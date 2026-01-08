function Install-Torch {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Version, # Package version

        [Parameter(Mandatory = $false)]
        [string]$Backend, # Backend. Example: auto

        [Parameter(Mandatory = $false)]
        [string]$IndexUrl, # Default index URL. Example: cu128

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

function Install-Package {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Id, # Package ID. Example: sdnq

        [Parameter(Mandatory = $true)]
        [string]$Version, # Package version

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    # We enforce exact package version for stability.
    $uvArgs = @("pip", "install", "$Id==$Version")

    Write-Debug "Installing $Id package with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $Id==$Version"
    }
}

function Install-Wheel {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Source, # Path to wheel file or URL

        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    $uvArgs = @("pip", "install", $Source)

    Write-Debug "Installing wheel from $Source with $Uv $uvArgs"
    & $Uv $uvArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $Source"
    }
}
