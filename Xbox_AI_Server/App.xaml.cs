using System;
using Windows.ApplicationModel;
using Windows.ApplicationModel.Activation;
using Windows.ApplicationModel.ExtendedExecution;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Navigation;

namespace Xbox_AI_Server
{
    /// <summary>
    /// UWP Application lifecycle manager for the David Xbox AI Server.
    ///
    /// Key design goal: The app must NEVER suspend. Suspension would kill the
    /// HttpListener, making the Xbox unreachable on the network. We use two
    /// mechanisms to prevent this:
    ///
    ///   1. ExtendedExecutionSession — Tells the OS we need to run indefinitely.
    ///   2. Visibility-change handling — Requests extended execution when the
    ///      app loses focus (e.g., user switches to Home or another game).
    /// </summary>
    sealed partial class App : Application
    {
        private ExtendedExecutionSession _extendedSession;

        public App()
        {
            this.InitializeComponent();
            this.Suspending += OnSuspending;
            this.EnteredBackground += OnEnteredBackground;
            this.LeavingBackground += OnLeavingBackground;
        }

        // ──────────────────────────────────────────────
        //  Launch
        // ──────────────────────────────────────────────

        protected override async void OnLaunched(LaunchActivatedEventArgs e)
        {
            Frame rootFrame = Window.Current.Content as Frame;

            if (rootFrame == null)
            {
                rootFrame = new Frame();
                rootFrame.NavigationFailed += OnNavigationFailed;
                Window.Current.Content = rootFrame;
            }

            if (e.PrelaunchActivated == false)
            {
                if (rootFrame.Content == null)
                {
                    rootFrame.Navigate(typeof(MainPage), e.Arguments);
                }

                Window.Current.Activate();
            }

            // Immediately request extended execution so the server stays alive
            await RequestExtendedExecutionAsync();
        }

        // ──────────────────────────────────────────────
        //  Extended Execution (Anti-Suspension)
        // ──────────────────────────────────────────────

        /// <summary>
        /// Requests an ExtendedExecutionSession with reason "Unspecified".
        /// This signals to the OS that the app should continue running
        /// even when it is no longer in the foreground.
        /// </summary>
        private async System.Threading.Tasks.Task RequestExtendedExecutionAsync()
        {
            // Dispose any existing session before creating a new one
            ClearExtendedSession();

            var session = new ExtendedExecutionSession
            {
                Reason = ExtendedExecutionReason.Unspecified,
                Description = "David AI Server must remain active to serve HTTP requests."
            };

            session.Revoked += ExtendedSession_Revoked;

            ExtendedExecutionResult result = await session.RequestExtensionAsync();

            if (result == ExtendedExecutionResult.Allowed)
            {
                _extendedSession = session;
                System.Diagnostics.Debug.WriteLine("[David] Extended execution GRANTED — server will not suspend.");
            }
            else
            {
                session.Dispose();
                System.Diagnostics.Debug.WriteLine("[David] Extended execution DENIED — app may suspend when minimized.");
            }
        }

        private void ExtendedSession_Revoked(object sender, ExtendedExecutionRevokedEventArgs args)
        {
            System.Diagnostics.Debug.WriteLine($"[David] Extended execution REVOKED. Reason: {args.Reason}");

            // If revoked due to system policy, try to re-acquire immediately
            if (args.Reason == ExtendedExecutionRevokedReason.SystemPolicy)
            {
                _ = RequestExtendedExecutionAsync();
            }
        }

        private void ClearExtendedSession()
        {
            if (_extendedSession != null)
            {
                _extendedSession.Revoked -= ExtendedSession_Revoked;
                _extendedSession.Dispose();
                _extendedSession = null;
            }
        }

        // ──────────────────────────────────────────────
        //  Background / Foreground transitions
        // ──────────────────────────────────────────────

        private async void OnEnteredBackground(object sender, EnteredBackgroundEventArgs e)
        {
            // Get a deferral so the system waits for us to finish
            var deferral = e.GetDeferral();

            System.Diagnostics.Debug.WriteLine("[David] Entered background — ensuring extended execution.");
            await RequestExtendedExecutionAsync();

            deferral.Complete();
        }

        private void OnLeavingBackground(object sender, LeavingBackgroundEventArgs e)
        {
            System.Diagnostics.Debug.WriteLine("[David] Returning to foreground.");
        }

        // ──────────────────────────────────────────────
        //  Standard UWP lifecycle
        // ──────────────────────────────────────────────

        private async void OnSuspending(object sender, SuspendingEventArgs e)
        {
            // Last-ditch attempt to stay alive
            var deferral = e.SuspendingOperation.GetDeferral();

            System.Diagnostics.Debug.WriteLine("[David] Suspension requested — attempting to block.");
            await RequestExtendedExecutionAsync();

            deferral.Complete();
        }

        private void OnNavigationFailed(object sender, NavigationFailedEventArgs e)
        {
            throw new Exception($"Failed to load Page: {e.SourcePageType.FullName}");
        }
    }
}
