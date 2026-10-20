param(
    [Parameter(Mandatory=$true)][string]$ConfigPath,
    [Parameter(Mandatory=$true)][ValidateSet('m3-norm','m3-norm-scatter','m3-stats','m3-viz','m3-load','m4-bench','m4-overfit','m4-train')][string]$Stage
)
# Redirect every cache/temp path to E: for this process only; no global settings change.
# native stderr is not fatal; the exit code decides
$config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$root=$config.root
$env:TEMP="$root/tmp"; $env:TMP=$env:TEMP
$env:PIP_CACHE_DIR="$root/cache/pip"; $env:CONDA_PKGS_DIRS="$root/cache/conda"
$env:TORCH_HOME="$root/cache/torch"; $env:MPLCONFIGDIR="$root/cache/matplotlib"
$env:APPDATA="$root/cache/appdata"; $env:LOCALAPPDATA="$root/cache/localappdata"
$env:CUDA_CACHE_PATH="$root/cache/cuda"; $env:TORCH_EXTENSIONS_DIR="$root/cache/torch_extensions"
$env:TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"; $env:XDG_CACHE_HOME="$root/cache"
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONNOUSERSITE='1'

$map=@{
    'm3-norm'         = @('m3_data.py','norm')
    'm3-norm-scatter' = @('m3_data.py','norm_scatter')
    'm3-stats'        = @('m3_data.py','stats')
    'm3-viz'          = @('m3_data.py','viz')
    'm3-load'         = @('m3_data.py','load')
    'm4-bench'        = @('m4_fno.py','bench')
    'm4-overfit'      = @('m4_fno.py','overfit')
    'm4-train'        = @('m4_fno.py','train')
}
$py,$st=$map[$Stage]
$run=Join-Path "$root/results" (("m34_"+$Stage+"_")+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $run | Out-Null
$log=Join-Path "$root/logs" ((Split-Path $run -Leaf)+'.log')
$timer=[Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference='Continue'
& "$root/envs/deepwave/Scripts/python.exe" -u "$PSScriptRoot/$py" $ConfigPath $st 2>&1 | Tee-Object -FilePath $log
$exitCode=$LASTEXITCODE
$timer.Stop()
@{stage=$Stage; process_wall_s=$timer.Elapsed.TotalSeconds; exit_code=$exitCode} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $run 'process.json')
if($exitCode -ne 0){throw "stage $Stage failed; preserve run and log: $run $log"}
Write-Output "RUN_DIR=$run"
Write-Output "LOG=$log"
