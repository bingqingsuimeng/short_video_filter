# -*- powershell -*-
# Prove the persisted env works end-to-end WITHOUT any manual export.
# Rebuilds the PATH exactly like a fresh Windows process sees it
# (Machine PATH + User PATH) and runs the real python import test.
$ErrorActionPreference = 'Stop'

$machine = [Environment]::GetEnvironmentVariable('PATH','Machine')
$user    = [Environment]::GetEnvironmentVariable('PATH','User')
$combined = "$machine;$user"

$env:PATH      = $combined
$env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH','User')
$env:CUDA_HOME = [Environment]::GetEnvironmentVariable('CUDA_HOME','User')

Write-Output "=== DLL resolution in fresh-env process ==="
foreach ($dll in @('nvinfer_11.dll','cudart64_13.dll','nvonnxparser_11.dll')) {
    $hit = where.exe $dll 2>$null | Select-Object -First 1
    if ($hit) { Write-Output ("  {0,-20} -> {1}" -f $dll, $hit) }
    else      { Write-Output ("  {0,-20} -> NOT FOUND" -f $dll) }
}

Write-Output "`n=== python import test (face_cleaner env) ==="
$py = 'D:\Users\Zz\miniconda3\envs\face_cleaner\python.exe'
& $py -c "import tensorrt as trt, pycuda.autoinit, pycuda.driver; print('TensorRT', trt.__version__); print('PYCUDA OK'); print('IMPORT SUCCESS')"
$code = $LASTEXITCODE
Write-Output "exit code: $code"
if ($code -ne 0) { throw "IMPORT TEST FAILED" }
