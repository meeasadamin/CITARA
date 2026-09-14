# Regenerate requirements.txt (Streamlit Community Cloud) from uv.lock.
# Replaces the torch pin with a direct CPU wheel URL: the +cpu build is not on PyPI, and an
# --extra-index-url makes uv pull other packages (numpy, ...) from the PyTorch index and fail.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$tmp = New-TemporaryFile
uv export --no-dev --no-hashes --no-emit-project --no-header --format requirements-txt -o $tmp.FullName | Out-Null

$lines = Get-Content $tmp.FullName
$cpuPin = $lines | Where-Object { $_ -match '^torch==([\d.]+)\+cpu' } | Select-Object -First 1
if (-not $cpuPin) { throw 'No torch==x.y.z+cpu pin found in uv export output.' }
$torchVersion = [regex]::Match($cpuPin, '^torch==([\d.]+)\+cpu').Groups[1].Value
$url = "https://download.pytorch.org/whl/cpu/torch-$torchVersion%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"

$out = foreach ($l in $lines) {
    if ($l -match '^torch==[\d.]+ ;') { continue }
    elseif ($l -match '^torch==[\d.]+\+cpu') { "torch @ $url ; sys_platform == 'linux' and platform_machine == 'x86_64'" }
    else { $l }
}
$header = @(
    '# Streamlit Community Cloud install file (Linux x86_64, Python 3.12). Local dev uses `uv sync`, not this file.',
    '# Regenerate with: scripts/export_requirements.ps1 (pins CPU-only torch by direct wheel URL).'
)
[IO.File]::WriteAllText((Join-Path (Get-Location) 'requirements.txt'), (($header + $out) -join "`n") + "`n")

# Verify it resolves for the deploy target (output file is discarded).
uv pip compile requirements.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 -q -o $tmp.FullName
$ok = $LASTEXITCODE -eq 0
Remove-Item $tmp.FullName
if (-not $ok) { throw 'requirements.txt does not resolve for the Streamlit Cloud target.' }
Write-Host "requirements.txt written (torch $torchVersion+cpu) and verified for linux/py3.12."
