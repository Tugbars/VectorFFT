# build_codelets.ps1 — compile the in-place AVX2 codelets into an in-repo object
# dir + response file, so the JIT (and the smoke test) link against a STABLE
# location instead of C:\tmp. Run once after checkout / when codelets change.
#
#   powershell -ExecutionPolicy Bypass -File jit\build_codelets.ps1
#
# Outputs (gitignored):  jit/generated/codelets/*.o  +  jit/generated/codelets.rsp
param([string]$Gcc = "C:\mingw152\mingw64\bin\gcc.exe")
$ErrorActionPreference = "Stop"
$jit  = $PSScriptRoot
$root = Split-Path $jit -Parent
$src  = Join-Path $root "codelets\inplace\avx2"
$out  = Join-Path $jit "generated\codelets"
$rsp  = Join-Path $jit "generated\codelets.rsp"
New-Item -ItemType Directory -Force $out | Out-Null

$cf = @("-O3","-mavx2","-mfma","-march=haswell","-Wno-incompatible-pointer-types","-Wno-unused-result")
$files = Get-ChildItem "$src\*.c"
$ok = 0; $fail = 0
foreach ($f in $files) {
    $o = Join-Path $out ($f.BaseName + ".o")
    & $Gcc @cf -c $f.FullName -o $o 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: $($f.Name)"; $fail++ } else { $ok++ }
}
# response file with forward-slash paths (gcc-friendly)
(Get-ChildItem "$out\*.o" | ForEach-Object { $_.FullName -replace '\\','/' }) |
    Set-Content $rsp -Encoding ascii
Write-Host "compiled $ok codelets ($fail failed) -> $out"
Write-Host "wrote $rsp ($((Get-Content $rsp).Count) entries)"
