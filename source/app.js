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
