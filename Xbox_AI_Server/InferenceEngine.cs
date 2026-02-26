using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Xbox_AI_Server
{
    /// <summary>
    /// InferenceEngine manages the full Phi-3 ONNX lifecycle:
    ///   1. Load model from disk via OnnxRuntimeGenAI (DirectML backend)
    ///   2. Tokenize prompts using the Phi-3 chat template
    ///   3. Generate response tokens with configurable search params
    ///   4. Decode and return the plain-text response
    ///
    /// This class is thread-safe for concurrent generation requests;
    /// OnnxRuntimeGenAI handles internal synchronization.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        private Model _model;
        private Tokenizer _tokenizer;
        private bool _isLoaded;
        private readonly object _loadLock = new object();

        // ──────────────────────────────────────────────
        //  Configuration
        // ──────────────────────────────────────────────

        /// <summary>Max tokens the model will generate per request.</summary>
        public int MaxLength { get; set; } = 1024;

        /// <summary>Sampling temperature (0.0 = greedy, 1.0 = creative).</summary>
        public float Temperature { get; set; } = 0.7f;

        /// <summary>Top-P nucleus sampling threshold.</summary>
        public float TopP { get; set; } = 0.9f;

        /// <summary>Whether the model has been loaded successfully.</summary>
        public bool IsLoaded => _isLoaded;

        // ──────────────────────────────────────────────
        //  Model Loading
        // ──────────────────────────────────────────────

        /// <summary>
        /// Loads the ONNX model and tokenizer from the specified directory.
        /// The directory must contain the model files and tokenizer config
        /// produced by the Phi-3 ONNX export (e.g., model.onnx, genai_config.json).
        /// </summary>
        public void LoadModel(string modelDirectoryPath)
        {
            lock (_loadLock)
            {
                if (_isLoaded)
                {
                    Debug.WriteLine("[InferenceEngine] Model already loaded — skipping.");
                    return;
                }

                if (!Directory.Exists(modelDirectoryPath))
                {
                    throw new DirectoryNotFoundException(
                        $"Model directory not found: {modelDirectoryPath}");
                }

                Debug.WriteLine($"[InferenceEngine] Loading model from: {modelDirectoryPath}");
                var sw = Stopwatch.StartNew();

                _model = new Model(modelDirectoryPath);
                _tokenizer = new Tokenizer(_model);

                sw.Stop();
                Debug.WriteLine($"[InferenceEngine] Model loaded in {sw.ElapsedMilliseconds} ms");

                _isLoaded = true;
            }
        }

        // ──────────────────────────────────────────────
        //  Inference
        // ──────────────────────────────────────────────

        /// <summary>
        /// Generates a response for the given user prompt using the Phi-3 model.
        /// Applies the standard Phi-3 chat template before tokenization.
        /// </summary>
        public async Task<InferenceResult> GenerateAsync(string userPrompt)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Model is not loaded. Call LoadModel() first.");

            return await Task.Run(() => GenerateSync(userPrompt));
        }

        /// <summary>
        /// Generates a response for the given user prompt, honouring max_tokens
        /// from the incoming request if provided.
        /// </summary>
        public async Task<InferenceResult> GenerateAsync(string userPrompt, int? maxTokens)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Model is not loaded. Call LoadModel() first.");

            int effectiveMaxLength = maxTokens.HasValue && maxTokens.Value > 0
                ? Math.Min(maxTokens.Value, 4096)
                : MaxLength;

            return await Task.Run(() => GenerateSync(userPrompt, effectiveMaxLength));
        }

        private InferenceResult GenerateSync(string userPrompt, int? overrideMaxLength = null)
        {
            var sw = Stopwatch.StartNew();

            // ── 1. Apply Phi-3 chat template ──
            string formattedPrompt = FormatPhi3Prompt(userPrompt);

            // ── 2. Tokenize ──
            var sequences = _tokenizer.Encode(formattedPrompt);
            int inputTokenCount = sequences[0].Length;

            // ── 3. Configure generation parameters ──
            using var generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", overrideMaxLength ?? MaxLength);
            generatorParams.SetSearchOption("temperature", Temperature);
            generatorParams.SetSearchOption("top_p", TopP);

            // ── 4. Generate tokens (v0.8+ Generator API) ──
            using var generator = new Generator(_model, generatorParams);
            generator.AppendTokenSequences(sequences);

            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
            }

            // ── 5. Decode output ──
            var outputSequence = generator.GetSequence(0);
            string fullOutput = _tokenizer.Decode(outputSequence);
            string assistantResponse = ExtractAssistantResponse(fullOutput, formattedPrompt);

            sw.Stop();

            // ── 6. Count generated tokens (output - input) ──
            int outputTokenCount = outputSequence.Length;
            int generatedTokens = outputTokenCount - inputTokenCount;

            Debug.WriteLine($"[InferenceEngine] Generated {generatedTokens} tokens in {sw.ElapsedMilliseconds} ms");

            return new InferenceResult
            {
                Text = assistantResponse.Trim(),
                TokensGenerated = generatedTokens,
                InferenceMs = sw.ElapsedMilliseconds
            };
        }

        // ──────────────────────────────────────────────
        //  Prompt Formatting
        // ──────────────────────────────────────────────

        /// <summary>
        /// Wraps the user prompt in the standard Phi-3 instruct template.
        ///
        /// Format:
        ///   <|user|>
        ///   {prompt}<|end|>
        ///   <|assistant|>
        /// </summary>
        private static string FormatPhi3Prompt(string userPrompt)
        {
            var sb = new StringBuilder();
            sb.AppendLine("<|user|>");
            sb.Append(userPrompt);
            sb.AppendLine("<|end|>");
            sb.Append("<|assistant|>");
            return sb.ToString();
        }

        /// <summary>
        /// Strips the echoed prompt prefix and any trailing special tokens
        /// from the decoded output to isolate the assistant's response.
        /// </summary>
        private static string ExtractAssistantResponse(string fullOutput, string formattedPrompt)
        {
            // The decoded output often starts with the input prompt echoed back.
            // Remove it to get only what the model generated.
            string response = fullOutput;

            if (response.StartsWith(formattedPrompt, StringComparison.Ordinal))
            {
                response = response.Substring(formattedPrompt.Length);
            }

            // Also try matching on the assistant tag alone (some decode paths
            // may not echo the full prompt verbatim).
            int assistantTagIndex = response.IndexOf("<|assistant|>", StringComparison.Ordinal);
            if (assistantTagIndex >= 0)
            {
                response = response.Substring(assistantTagIndex + "<|assistant|>".Length);
            }

            // Strip trailing end token if present
            int endTagIndex = response.IndexOf("<|end|>", StringComparison.Ordinal);
            if (endTagIndex >= 0)
            {
                response = response.Substring(0, endTagIndex);
            }

            return response;
        }

        // ──────────────────────────────────────────────
        //  Disposal
        // ──────────────────────────────────────────────

        public void Dispose()
        {
            _tokenizer?.Dispose();
            _model?.Dispose();
            _isLoaded = false;

            Debug.WriteLine("[InferenceEngine] Disposed.");
        }
    }

    /// <summary>
    /// Encapsulates the result of a single inference call.
    /// </summary>
    public class InferenceResult
    {
        /// <summary>The decoded text generated by the model.</summary>
        public string Text { get; set; }

        /// <summary>Number of new tokens generated (excludes input tokens).</summary>
        public int TokensGenerated { get; set; }

        /// <summary>Wall-clock time for the full inference pipeline in ms.</summary>
        public long InferenceMs { get; set; }
    }
}
