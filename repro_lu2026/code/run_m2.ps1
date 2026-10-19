param(
    [Parameter(Mandatory=$true)][string]$ConfigPath, [ValidateSet("pilot","bulk")][string]$Stage="pilot"
)
$ErrorActionPreference='Stop'
$config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$root=$config.root
$env:TEMP="$root/tmp"; $env:TMP=$env:TEMP
$env:PIP_CACHE_DIR="$root/cache/pip"; $env:CONDA_PKGS_DIRS="$root/cache/conda"
$env:TORCH_HOME="$root/cache/torch"; $env:MPLCONFIGDIR="$root/cache/matplotlib"
$env:APPDATA="$root/cache/appdata"; $env:LOCALAPPDATA="$root/cache/localappdata"
$env:CUDA_CACHE_PATH="$root/cache/cuda"; $env:TORCH_EXTENSIONS_DIR="$root/cache/torch_extensions"
$env:TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"; $env:XDG_CACHE_HOME="$root/cache"
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONNOUSERSITE='1'
$run=Join-Path "$root/results" ("m2_session_"+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $run | Out-Null
$log=Join-Path "$root/logs" ((Split-Path $run -Leaf)+'.log')
$timer=[Diagnostics.Stopwatch]::StartNew()
& "$root/envs/deepwave/Scripts/python.exe" -u "$PSScriptRoot/generate_m2.py" $ConfigPath $Stage 2>&1 | Tee-Object -FilePath $log
$exitCode=$LASTEXITCODE
$timer.Stop()
@{process_wall_s=$timer.Elapsed.TotalSeconds; exit_code=$exitCode; includes_analysis_and_plots=$true} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $run 'process.json')
if($exitCode -ne 0){throw "Deepwave failed; preserve run and log: $run $log"}
Write-Output "RUN_DIR=$run"
