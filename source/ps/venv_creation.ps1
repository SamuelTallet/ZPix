function New-VirtualEnv {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Python, # Python version

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    & $Uv venv --python $Python --clear --force

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create venv"
    }
}
