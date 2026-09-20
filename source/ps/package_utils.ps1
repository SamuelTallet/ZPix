function Install-Dependency {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Spec, # Specifier (package, wheel, etc.)

        [Parameter(Mandatory = $false)]
        [string]$Backend, # PyTorch backend. Example: auto

        [Parameter(Mandatory = $false)]
        [string]$IndexUrl, # Default index URL. Example: https://download.pytorch.org/whl/cu130

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    $uvArgs = @("pip", "install", $Spec)

    if ($Backend) {
        $uvArgs += "--torch-backend=$Backend"
    }

    if ($IndexUrl) {
        $uvArgs += "--default-index"
        $uvArgs += $IndexUrl
    }

    Write-Debug "Installing dependency with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependency $Spec"
    }
}

function Install-Requirements {
    param (
        [Parameter(Mandatory = $true)]
        [string]$File, # Path to requirements*.txt

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    $uvArgs = @("pip", "install", "-r", $File)

    Write-Debug "Installing requirements with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dependencies from $File"
    }
}
