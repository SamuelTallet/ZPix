/*
 * Z-Image Turbo Gradio app custom JavaScript.
 */

// Delegate events for some elements.
document.addEventListener("click", (event) => {
    if (event.target.closest("#swap-lora-btn")) {
        return swapLora()
    }

    /** @type {HTMLAnchorElement | null} */
    const link = event.target.closest("a")

    if (link && link.target === "_blank") {
        openExternalLink(link, event)
    }
})

/**
 * Open a link in default browser.
 *
 * @param {HTMLAnchorElement} link
 * @param {Event} event
 */
function openExternalLink(link, event) {
    event.preventDefault() // Instead of opening a new webview
    // thanks to a custom binding. See webview.cpp
    window.openWithDefaultBrowser(link.href)
}

/**
 * Swap LoRA.
 */
async function swapLora() {
    /**
     * Absolute path to LoRA file selected by user.
     * @type {string}
     */
    const path = await window.openNativeFileDialog()
    // This avoids an unnecessary local upload. See webview.cpp

    if (!path) return // When the user cancels.

    /** @type {HTMLTextAreaElement} */
    const portal = document.querySelector("#lora-path textarea")

    // We place the LoRA path in a "portal" hidden textarea.
    // This "portal" transfers data and control to the backend.
    // Appending a timestamp forces the change event to fire
    // when the user loads-unloads-reloads the same LoRA file.
    portal.value = `${path}|${Math.floor(Date.now() / 1000)}`
    portal.dispatchEvent(new Event("input"))
}