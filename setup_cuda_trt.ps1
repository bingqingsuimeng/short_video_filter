# -*- powershell -*-
# Make CUDA 13.1 + TensorRT 11.3 available to ALL new processes (User scope).
# Idempotent: safe to run repeatedly. No admin needed (User scope).
#
$ErrorActionPreference = 'Stop'

$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
$trtRoot  = 'C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-11.3.0.99'

$entries = @(
    (Join-Path $cudaRoot 'bin'),
    (Join-Path $cudaRoot 'libnvvp'),
    (Join-Path $trtRoot  'bin')
)

# keep only dirs that actually exist (libnvvp is optional / may be absent)
$entries = @($entries | Where-Object { Test-Path $_ })

# ---- PATH (User) ----
$userPath = [Environment]::GetEnvironmentVariable('PATH','User')
if (-not $userPath) { $userPath = '' }
$parts = @($userPath -split ';' | Where-Object { $_.Trim() -ne '' })
$changed = $false
foreach ($e in $entries) {
    $exists = $parts | Where-Object { $_.Trim().ToLower() -eq $e.ToLower() }
    if (-not $exists) {
        Write-Output "  + PATH: $e"
        $parts = @($e) + $parts   # prepend
        $changed = $true
    }
}
if ($changed) {
    [Environment]::SetEnvironmentVariable('PATH', ($parts -join ';'), 'User')
} else {
    Write-Output "  PATH already contains all CUDA/TRT entries."
}

# ---- CUDA_PATH / CUDA_HOME (User) ----
[Environment]::SetEnvironmentVariable('CUDA_PATH', $cudaRoot, 'User')
[Environment]::SetEnvironmentVariable('CUDA_HOME', $cudaRoot, 'User')
Write-Output "  CUDA_PATH = $cudaRoot"
Write-Output "  CUDA_HOME = $cudaRoot"

Write-Output "DONE. New terminals / Claude Code launches will inherit these."
