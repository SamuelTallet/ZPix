function Invoke-App {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Model, # Source model to load

        [Parameter(Mandatory = $true)]
        [string]$BackupModel, # Backup model if needed

        [Parameter(Mandatory = $true)]
        [int]$Port, # Local port to use

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    # Current locale. Example: fr-FR
    $locale = (Get-WinSystemLocale).Name

    $uvArgs = @(
        "run", "app.py",
        "--model", $Model,
        "--backup-model", $BackupModel,
        "--port", $Port,
        "--locale", $locale
    )

    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to run app"
    }
}
