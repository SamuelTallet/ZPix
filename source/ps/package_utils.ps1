function Install-Torch {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Version, # Package version

        [Parameter(Mandatory = $false)]
        [string]$Backend, # Backend. Example: auto

        [Parameter(Mandatory = $false)]
        [string]$IndexUrl, # Default index URL. Example: cu130

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    # We enforce exact package version for stability.
    $uvArgs = @("pip", "install", "torch==$Version")

    if ($Backend) {
        $uvArgs += "--torch-backend=$Backend"
    }

    if ($IndexUrl) {
        $uvArgs += "--default-index"
        $uvArgs += "https://download.pytorch.org/whl/$IndexUrl"
    }

    Write-Debug "Installing torch package with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install torch==$Version"
    }
}

function Install-TorchVision {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Version, # Package version

        [Parameter(Mandatory = $false)]
        [string]$Backend, # Backend. Example: auto

        [Parameter(Mandatory = $false)]
        [string]$IndexUrl, # Default index URL. Example: cu130

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    # We enforce exact package version for stability.
    $uvArgs = @("pip", "install", "torchvision==$Version")

    if ($Backend) {
        $uvArgs += "--torch-backend=$Backend"
    }

    if ($IndexUrl) {
        $uvArgs += "--default-index"
        $uvArgs += "https://download.pytorch.org/whl/$IndexUrl"
    }

    Write-Debug "Installing torchvision package with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install torchvision==$Version"
    }
}

function Install-Dependency {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Spec, # Specifier (package, wheel, etc.)

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    $uvArgs = @("pip", "install", $Spec)

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
