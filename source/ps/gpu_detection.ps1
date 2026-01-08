class Gpu {
    [string]$Name # Example: NVIDIA GeForce RTX 3070 Laptop GPU
    [int64]$Memory # Example: 8589934592 (bytes)
    [string]$Vendor # Examples: NVIDIA, AMD, Intel
}

function Get-Gpus {
    [OutputType([Gpu[]])]
    param()

    $gpus = @()

    # Device class for display drivers (and video miniport drivers).
    # See: https://learn.microsoft.com/en-us/windows-hardware/drivers/install/system-defined-device-setup-classes-available-to-vendors
    $deviceClass = "4d36e968-e325-11ce-bfc1-08002be10318"

    # Path to registry keys containing display adapters information.
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{$deviceClass}\0*"

    $adapters = Get-ItemProperty -Path $regPath -ErrorAction SilentlyContinue

    foreach ($adapter in $adapters) {
        $name = $adapter."HardwareInformation.AdapterString"

        if (-not $name) {
            Write-Debug "Failed to get name of a display adapter."
            continue
        }

        # Decode adapter name if it's stored as a byte array.
        if ($name -is [byte[]]) {
            $name = [System.Text.Encoding]::Unicode.GetString($name)
        }

        $memory = $adapter."HardwareInformation.qwMemorySize"

        # Fallback to MemorySize if qwMemorySize is missing.
        if (-not $memory -and
            ($altMemory = $adapter."HardwareInformation.MemorySize") -and
            ($altMemory -is [byte[]])
        ) {
            $memory = [BitConverter]::ToInt32($altMemory, 0)
        }

        if (-not $memory) {
            Write-Debug "Failed to get memory size of $name."
            continue
        }

        $vendor = switch -Regex ($adapter.MatchingDeviceId) {
            "ven_10de" { "NVIDIA" }
            "ven_1002" { "AMD" }
            "ven_8086" { "Intel" }
            Default { "Unknown" }
        }

        if ($vendor -eq "Unknown") {
            Write-Debug "Failed to get vendor name of $name."
            continue
        }

        $gpus += [Gpu]@{
            Name   = $name
            Memory = $memory
            Vendor = $vendor
        }
    }

    return $gpus
}
