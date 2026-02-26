using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Windows.ApplicationModel;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;

namespace Xbox_AI_Server
{
    /// <summary>
    /// MainPage hosts the HTTP server and routes prompts to the InferenceEngine.
    /// POST /api/prompt  →  Runs Phi-3 inference and returns the AI response.
    /// GET  /health      →  Returns server + model status.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private HttpListener _listener;
        private CancellationTokenSource _cts;
        private InferenceEngine _engine;
        private int _requestCount;
        private bool _modelReady;

        public MainPage()
        {
            this.InitializeComponent();
            this.Loaded += MainPage_Loaded;
        }

        // ──────────────────────────────────────────────
        //  Lifecycle
        // ──────────────────────────────────────────────

        private async void MainPage_Loaded(object sender, RoutedEventArgs e)
        {
            // Step 1: Load the ONNX model
            await LoadModelAsync();

            // Step 2: Start the HTTP server
            await StartServerAsync();
        }

        // ──────────────────────────────────────────────
        //  Model Loading
        // ──────────────────────────────────────────────

        private async Task LoadModelAsync()
        {
            await UpdateStatusAsync("⏳ Loading Phi-3 model...");

            try
            {
                _engine = new InferenceEngine();

                // Resolve model path relative to the installed app package
                string modelPath = Path.Combine(
                    Package.Current.InstalledLocation.Path,
                    "Assets", "Model", "directml", "directml-int4-awq-block-128");

                // Load on a background thread (can take 10-30s on Xbox)
                await Task.Run(() => _engine.LoadModel(modelPath));

                _modelReady = true;
                await UpdateStatusAsync("🧠 Phi-3 model loaded. Starting server...");
            }
            catch (Exception ex)
            {
                _modelReady = false;
                await UpdateStatusAsync($"❌ Model load failed: {ex.Message}");
                Debug.WriteLine($"[MainPage] Model load error: {ex}");
            }
        }

        // ──────────────────────────────────────────────
        //  HTTP Server
        // ──────────────────────────────────────────────

        private async Task StartServerAsync()
        {
            _cts = new CancellationTokenSource();

            try
            {
                _listener = new HttpListener();
                _listener.Prefixes.Add("http://*:8080/");
                _listener.Start();

                string status = _modelReady
                    ? "✅ Server LIVE on port 8080 — Phi-3 ready"
                    : "⚠️ Server LIVE on port 8080 — Model NOT loaded (fallback mode)";

                await UpdateStatusAsync(status);

                // Begin the accept loop on a background thread
                await Task.Run(() => AcceptLoopAsync(_cts.Token));
            }
            catch (HttpListenerException ex)
            {
                await UpdateStatusAsync($"❌ Failed to start: {ex.Message}");
            }
        }

        private async Task AcceptLoopAsync(CancellationToken token)
        {
            while (!token.IsCancellationRequested && _listener.IsListening)
            {
                try
                {
                    var context = await _listener.GetContextAsync();
                    _ = Task.Run(() => HandleRequestAsync(context));
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (HttpListenerException)
                {
                    break;
                }
            }
        }

        // ──────────────────────────────────────────────
        //  Request Handling
        // ──────────────────────────────────────────────

        private async Task HandleRequestAsync(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;

            try
            {
                // ── Health Check ──
                if (request.HttpMethod == "GET" && request.Url.AbsolutePath == "/health")
                {
                    await WriteJsonResponseAsync(response, 200, new
                    {
                        status = "healthy",
                        modelLoaded = _modelReady,
                        requestsServed = _requestCount
                    });
                    return;
                }

                // ── Prompt Endpoint ──
                if (request.HttpMethod == "POST" && request.Url.AbsolutePath == "/api/prompt")
                {
                    await HandlePromptAsync(request, response);
                    return;
                }

                // ── 404 ──
                await WriteJsonResponseAsync(response, 404,
                    new { error = "Not found. Use POST /api/prompt or GET /health." });
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[MainPage] Request error: {ex}");
                await WriteJsonResponseAsync(response, 500,
                    new { error = ex.Message });
            }
        }

        private async Task HandlePromptAsync(HttpListenerRequest request, HttpListenerResponse response)
        {
            // ── Read body ──
            string body;
            using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            {
                body = await reader.ReadToEndAsync();
            }

            // ── Deserialize ──
            PromptRequest promptRequest;
            try
            {
                promptRequest = JsonSerializer.Deserialize<PromptRequest>(body,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            }
            catch (JsonException)
            {
                await WriteJsonResponseAsync(response, 400,
                    new { error = "Invalid JSON. Expected: { \"prompt\": \"...\" }" });
                return;
            }

            if (string.IsNullOrWhiteSpace(promptRequest?.Prompt))
            {
                await WriteJsonResponseAsync(response, 400,
                    new { error = "Missing required field: 'prompt'." });
                return;
            }

            // ── Run inference or fallback ──
            PromptResponse result;

            if (_modelReady && _engine != null)
            {
                try
                {
                    var inferenceResult = await _engine.GenerateAsync(
                        promptRequest.Prompt,
                        promptRequest.MaxTokens > 0 ? promptRequest.MaxTokens : (int?)null);

                    result = new PromptResponse
                    {
                        Response = inferenceResult.Text,
                        TokensUsed = inferenceResult.TokensGenerated,
                        InferenceMs = inferenceResult.InferenceMs
                    };
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[MainPage] Inference error: {ex}");
                    await WriteJsonResponseAsync(response, 500,
                        new { error = $"Inference failed: {ex.Message}" });
                    return;
                }
            }
            else
            {
                // Fallback: model not loaded — return echo so the network layer
                // can still be tested end-to-end
                result = new PromptResponse
                {
                    Response = $"[FALLBACK] Model not loaded. Echo: {promptRequest.Prompt}",
                    TokensUsed = 0,
                    InferenceMs = 0
                };
            }

            Interlocked.Increment(ref _requestCount);
            await UpdateRequestCountAsync();

            await WriteJsonResponseAsync(response, 200, result);
        }

        // ──────────────────────────────────────────────
        //  Helpers
        // ──────────────────────────────────────────────

        private async Task WriteJsonResponseAsync(HttpListenerResponse response, int statusCode, object payload)
        {
            response.StatusCode = statusCode;
            response.ContentType = "application/json";

            var json = JsonSerializer.Serialize(payload,
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });

            var buffer = Encoding.UTF8.GetBytes(json);
            response.ContentLength64 = buffer.Length;

            await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            response.OutputStream.Close();
        }

        private async Task UpdateStatusAsync(string message)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                StatusText.Text = message;
            });
        }

        private async Task UpdateRequestCountAsync()
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                RequestCountText.Text = $"Requests served: {_requestCount}";
            });
        }
    }

    // ──────────────────────────────────────────────
    //  JSON Models
    // ──────────────────────────────────────────────

    public class PromptRequest
    {
        public string Prompt { get; set; }
        public int MaxTokens { get; set; } = 256;
    }

    public class PromptResponse
    {
        public string Response { get; set; }
        public int TokensUsed { get; set; }
        public long InferenceMs { get; set; }
    }
}
