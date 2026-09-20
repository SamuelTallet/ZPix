function Get-CudaVersion {
    [OutputType([string])] # Example: 13.0
    param (
        [Parameter(Mandatory = $true)]
        [string]$Uv # Path to uv executable
    )

    $uvArgs = @("run", "python", "-c", "import torch; print(torch.version.cuda)")
    $cudaVersion = "$(& $Uv $uvArgs)".Trim()

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to get PyTorch CUDA version"
    }

    # `torch.version.cuda` is `None` when PyTorch was built without CUDA support.
    if ($cudaVersion -eq "None") {
        return $null
    }

    return $cudaVersion
}
