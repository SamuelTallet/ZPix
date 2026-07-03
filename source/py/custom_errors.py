"""Custom exceptions."""


class EventAbort(Exception):
    """Silently aborts a Gradio event chain.

    Raised from a validator step to shunt the chain into its `.failure`
    branch without surfacing anything to the user. Because the app launches
    with `show_error=False`, Gradio's `error_payload` returns
    `{"error": None}`, so this exception is only logged to the console.
    """
