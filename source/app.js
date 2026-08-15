/*
 * ZPix Gradio app custom JavaScript.
 */

// Delegate events to work with Gradio rendering.
document.addEventListener("mouseover", (event) => {
    if (event.target.closest("#prompt textarea")) {
        const prompt = document.getElementById("prompt")
        if (prompt?.title) hidePromptTooltip(prompt)
    }
})
document.addEventListener("mouseleave", (event) => {
    if (event.target.closest?.("#prompt textarea")) {
        const prompt = document.getElementById("prompt")
        if (prompt?.dataset.title) restorePromptTooltip(prompt)
    }
}, true) // Required because mouseleave doesn't bubble.

// A refresh can close the server (see app.unload)
// or disturb the app state.
if (performance.getEntriesByType("navigation")[0]?.type === "reload") {
    const notice = document.createElement("div")
    notice.className = "refresh-notice"
    notice.textContent = "Page reload broke app."
    notice.textContent += " Close this tab or window and run ZPix again."
    document.body.replaceChildren(notice)
}

/**
 * Hide the prompt tooltip on its textarea mouseover
 * so tooltip doesn't cause inconvenience to the user.
 * @param {HTMLElement} prompt
 */
function hidePromptTooltip(prompt) {
    prompt.dataset.title = prompt.title
    prompt.title = ""
}

/**
 * Restore the prompt tooltip on its textarea mouseleave.
 * @param {HTMLElement} prompt
 */
function restorePromptTooltip(prompt) {
    prompt.title = prompt.dataset.title
    delete prompt.dataset.title
}
