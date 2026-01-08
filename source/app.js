/*
 * App custom JavaScript.
 */

// Opens external links with default browser:
document.addEventListener("click", (event) => {
    const link = event.target.closest("a")
    if (link && link.target === "_blank") {
        event.preventDefault()
        // Custom binding 👇 See webview.cpp
        window.openWithDefaultBrowser(link.href)
    }
})
