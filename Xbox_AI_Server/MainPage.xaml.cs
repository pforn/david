using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;

namespace Xbox_AI_Server
{
    /// <summary>
    /// MainPage hosts a lightweight HTTP server on port 8080.
    /// POST /api/prompt  →  Accepts a JSON prompt and returns a static response.
    /// GET  /health      →  Returns 200 OK for health-check pings.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private HttpListener _listener;
        private CancellationTokenSource _cts;
        private int _requestCount;

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
            await StartServerAsync();
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

                await UpdateStatusAsync("✅ Server LIVE on port 8080");

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
                    // Wait for an incoming request
                    var context = await _listener.GetContextAsync();
                    _ = Task.Run(() => HandleRequestAsync(context));
                }
                catch (ObjectDisposedException)
                {
                    break; // Listener was shut down
                }
                catch (HttpListenerException)
                {
                    break; // Listener error, bail out
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
                    await WriteJsonResponseAsync(response, 200,
                        new { status = "healthy", uptime = "ok" });
                    return;
                }

                // ── Prompt Endpoint ──
                if (request.HttpMethod == "POST" && request.Url.AbsolutePath == "/api/prompt")
                {
                    await HandlePromptAsync(request, response);
                    return;
                }

                // ── 404 for everything else ──
                await WriteJsonResponseAsync(response, 404,
                    new { error = "Not found. Use POST /api/prompt or GET /health." });
            }
            catch (Exception ex)
            {
                await WriteJsonResponseAsync(response, 500,
                    new { error = ex.Message });
            }
        }

        private async Task HandlePromptAsync(HttpListenerRequest request, HttpListenerResponse response)
        {
            // Read the incoming JSON body
            string body;
            using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            {
                body = await reader.ReadToEndAsync();
            }

            // Deserialize and validate
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

            // ── Static response for Step 1 (no ONNX yet) ──
            var result = new PromptResponse
            {
                Response = "Connection successful. Xbox API is live.",
                TokensUsed = 0,
                InferenceMs = 0
            };

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
    //  JSON Models (inline for now; will extract later)
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
