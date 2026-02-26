using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace Xbox_AI_Server
{
    /// <summary>
    /// InferenceEngine manages the full Phi-3 ONNX lifecycle using the base
    /// OnnxRuntime (no GenAI layer) to stay within Xbox UWP sandbox constraints:
    ///
    ///   1. Load model.onnx via InferenceSession with DirectML execution provider
    ///   2. Load tokenizer via Microsoft.ML.Tokenizers (BPE from tokenizer.json)
    ///   3. Run a manual greedy autoregressive decoding loop
    ///   4. Decode output token IDs back to plain text
    ///
    /// All memory is managed (.NET heap only). No unsafe blocks, no Win32 P/Invoke.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        // ── ONNX tensor / model names for Phi-3 Mini ONNX export ──────────────
        // These match the standard Hugging Face Optimum ONNX export of Phi-3 Mini.
        // Verify with: onnxruntime Python → sess.get_inputs() / sess.get_outputs()
        private const string InputIdsName        = "input_ids";
        private const string AttentionMaskName   = "attention_mask";
        private const string LogitsOutputName    = "logits";

        // Phi-3 Mini EOS token ID — matches `<|end|>` in its tokenizer.
        // Value 32007 is the standard EOS for the Phi-3 instruct chat template.
        private const int EosTokenId = 32007;

        private InferenceSession _session;
        private Tokenizer        _tokenizer;
        private bool             _isLoaded;
        private readonly object  _loadLock = new object();

        // ──────────────────────────────────────────────
        //  Configuration
        // ──────────────────────────────────────────────

        /// <summary>Max new tokens the model will generate per request.</summary>
        public int MaxLength { get; set; } = 1024;

        /// <summary>Whether the model and tokenizer have been loaded successfully.</summary>
        public bool IsLoaded => _isLoaded;

        // ──────────────────────────────────────────────
        //  Model Loading
        // ──────────────────────────────────────────────

        /// <summary>
        /// Loads the ONNX model and BPE tokenizer from the specified directory.
        /// The directory must contain:
        ///   - model.onnx          (the ONNX export of Phi-3 Mini)
        ///   - tokenizer.json      (HuggingFace BPE vocab/merges)
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
                    throw new DirectoryNotFoundException(
                        $"Model directory not found: {modelDirectoryPath}");

                string modelPath     = Path.Combine(modelDirectoryPath, "model.onnx");
                string tokenizerPath = Path.Combine(modelDirectoryPath, "tokenizer.json");

                if (!File.Exists(modelPath))
                    throw new FileNotFoundException(
                        $"model.onnx not found in: {modelDirectoryPath}", modelPath);

                if (!File.Exists(tokenizerPath))
                    throw new FileNotFoundException(
                        $"tokenizer.json not found in: {modelDirectoryPath}", tokenizerPath);

                Debug.WriteLine($"[InferenceEngine] Loading model from: {modelDirectoryPath}");
                var sw = Stopwatch.StartNew();

                // ── 1. Session options — DirectML EP (GPU acceleration on Xbox) ──
                var sessionOptions = new SessionOptions();
                sessionOptions.AppendExecutionProvider_DML(deviceId: 0);

                // ── 2. Load ONNX model ──
                _session = new InferenceSession(modelPath, sessionOptions);
                Debug.WriteLine($"[InferenceEngine] InferenceSession created in {sw.ElapsedMilliseconds} ms");

                // ── 3. Load BPE tokenizer ──
                using (var tokenizerStream = File.OpenRead(tokenizerPath))
                {
                    _tokenizer = BpeTokenizer.Create(tokenizerStream);
                }
                Debug.WriteLine($"[InferenceEngine] Tokenizer loaded in {sw.ElapsedMilliseconds} ms");

                sw.Stop();
                _isLoaded = true;
                Debug.WriteLine($"[InferenceEngine] Total load time: {sw.ElapsedMilliseconds} ms");
            }
        }

        // ──────────────────────────────────────────────
        //  Inference (Public API — same contract as before)
        // ──────────────────────────────────────────────

        /// <summary>
        /// Generates a response for the given user prompt using the Phi-3 model.
        /// Applies the standard Phi-3 chat template before tokenization.
        /// </summary>
        public async Task<InferenceResult> GenerateAsync(string userPrompt)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Model is not loaded. Call LoadModel() first.");

            return await Task.Run(() => GenerateSync(userPrompt, null));
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

        // ──────────────────────────────────────────────
        //  Core Inference — Manual Greedy Decode Loop
        // ──────────────────────────────────────────────

        private InferenceResult GenerateSync(string userPrompt, int? overrideMaxLength)
        {
            var sw = Stopwatch.StartNew();
            int maxNewTokens = overrideMaxLength ?? MaxLength;

            // ── 1. Apply Phi-3 chat template and tokenize ──
            string formattedPrompt = FormatPhi3Prompt(userPrompt);
            var    encoding        = _tokenizer.Encode(formattedPrompt);
            var    promptTokenIds  = encoding.Ids;

            // Build the running token list (prompt + generated tokens)
            var tokenIds = new List<int>(promptTokenIds);
            int inputLen = tokenIds.Count;

            Debug.WriteLine($"[InferenceEngine] Prompt tokenized to {inputLen} tokens. Running greedy decode...");

            int generatedCount = 0;

            // ── 2. Greedy autoregressive loop ──
            for (int step = 0; step < maxNewTokens; step++)
            {
                int seqLen = tokenIds.Count;

                // Build input tensors — shape [1, seqLen]
                var inputIdsTensor    = new DenseTensor<long>(new[] { 1, seqLen });
                var attentionMaskTensor = new DenseTensor<long>(new[] { 1, seqLen });

                for (int i = 0; i < seqLen; i++)
                {
                    inputIdsTensor[0, i]      = (long)tokenIds[i];
                    attentionMaskTensor[0, i] = 1L;
                }

                // Build NamedOnnxValue inputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(InputIdsName,      inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor(AttentionMaskName, attentionMaskTensor),
                };

                // ── 3. Run the session ──
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs;
                try
                {
                    outputs = _session.Run(inputs);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[InferenceEngine] Session.Run failed at step {step}: {ex.Message}");
                    throw;
                }

                using (outputs)
                {
                    // ── 4. Extract logits — shape [1, seqLen, vocabSize] ──
                    var logitsTensor = outputs.First(o => o.Name == LogitsOutputName)
                                             .AsEnumerable<float>()
                                             .ToArray();

                    int vocabSize = logitsTensor.Length / seqLen;

                    // Logits for the LAST position only (autoregressive next-token)
                    int lastTokenOffset = (seqLen - 1) * vocabSize;

                    // ── 5. Greedy argmax ──
                    int nextTokenId = ArgMax(logitsTensor, lastTokenOffset, vocabSize);

                    // ── 6. Append and check for EOS ──
                    tokenIds.Add(nextTokenId);
                    generatedCount++;

                    if (nextTokenId == EosTokenId)
                    {
                        Debug.WriteLine($"[InferenceEngine] EOS hit at step {step + 1}.");
                        break;
                    }
                }
            }

            sw.Stop();
            Debug.WriteLine($"[InferenceEngine] Generated {generatedCount} tokens in {sw.ElapsedMilliseconds} ms");

            // ── 7. Decode generated tokens (exclude prompt tokens) ──
            var generatedIds = tokenIds.Skip(inputLen).ToArray();
            string decodedText = _tokenizer.Decode(generatedIds) ?? string.Empty;
            string assistantResponse = StripSpecialTokens(decodedText).Trim();

            return new InferenceResult
            {
                Text            = assistantResponse,
                TokensGenerated = generatedCount,
                InferenceMs     = sw.ElapsedMilliseconds
            };
        }

        // ──────────────────────────────────────────────
        //  Helpers
        // ──────────────────────────────────────────────

        /// <summary>
        /// Finds the index of the maximum value within a slice of the array.
        /// Used for greedy decoding (argmax over vocabulary logits).
        /// </summary>
        private static int ArgMax(float[] array, int offset, int length)
        {
            int   bestIdx   = 0;
            float bestVal   = float.MinValue;

            for (int i = 0; i < length; i++)
            {
                float val = array[offset + i];
                if (val > bestVal)
                {
                    bestVal = val;
                    bestIdx = i;
                }
            }

            return bestIdx;
        }

        /// <summary>
        /// Wraps the user prompt in the Phi-3 instruct chat template.
        /// Format: &lt;|user|&gt;\n{prompt}&lt;|end|&gt;\n&lt;|assistant|&gt;
        /// </summary>
        private static string FormatPhi3Prompt(string userPrompt)
        {
            var sb = new StringBuilder();
            sb.Append("<|user|>\n");
            sb.Append(userPrompt);
            sb.Append("<|end|>\n");
            sb.Append("<|assistant|>\n");
            return sb.ToString();
        }

        /// <summary>
        /// Strips the echoed prompt prefix and any trailing special tokens
        /// from the decoded output to isolate the assistant's response.
        /// </summary>
        private static string StripSpecialTokens(string fullOutput)
        {
            string response = fullOutput;

            // Try matching on the assistant tag
            int assistantTagIndex = response.IndexOf("<|assistant|>", StringComparison.Ordinal);
            if (assistantTagIndex >= 0)
            {
                response = response.Substring(assistantTagIndex + "<|assistant|>".Length).TrimStart();
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
            _session?.Dispose();
            _isLoaded = false;

            System.Diagnostics.Debug.WriteLine("[InferenceEngine] Disposed.");
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
