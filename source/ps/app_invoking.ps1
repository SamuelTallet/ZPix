function Invoke-App {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    # Current locale. Example: fr-FR
    $locale = (Get-WinSystemLocale).Name

    $uvArgs = @(
        "run", "app.py",
        # No --in-browser arg here since on Windows
        # the app is shown in a WebView2 window.
        "--locale", $locale
    )

    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to run app"
    }
}
