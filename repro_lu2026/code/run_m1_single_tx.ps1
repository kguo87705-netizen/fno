param([string]$ConfigPath = (Join-Path $PSScriptRoot '..\configs\m1_single_tx.json'))
$ErrorActionPreference = 'Stop'
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
$cfg = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$root = $cfg.paths.root
if ($root -ne 'E:\FNO_research') { throw 'Unexpected project root' }
$env:TEMP = "$root\tmp"
$env:TMP = $env:TEMP
$env:MATLAB_PREFDIR = "$root\matlab_prefs"
$env:APPDATA = "$root\cache\appdata"
$env:LOCALAPPDATA = "$root\cache\localappdata"
$env:CUDA_CACHE_PATH = "$root\cache\cuda"
$env:PIP_CACHE_DIR = "$root\cache\pip"
$env:CONDA_PKGS_DIRS = "$root\cache\conda"
$env:TORCH_HOME = "$root\cache\torch"
$env:MPLCONFIGDIR = "$root\cache\matplotlib"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:XDG_CACHE_HOME = "$root\cache"
$hash = (Get-FileHash -LiteralPath $cfg.paths.label_npy -Algorithm SHA256).Hash
if ($hash -ne $cfg.paths.label_sha256) { throw 'Label checksum mismatch' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$outdir = "$root\results\m1_single_tx_$stamp"
if (Test-Path -LiteralPath $outdir) { throw 'Refusing overwrite of run directory' }
New-Item -ItemType Directory -Path $outdir | Out-Null
$log = "$root\logs\m1_single_tx_$stamp.log"
$csv = "$root\logs\m1_gpu_$stamp.csv"
Copy-Item -LiteralPath $ConfigPath -Destination "$outdir\config.json"
$smi = (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
$baseline = & $smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,nounits
$baseline | Set-Content -LiteralPath "$outdir\gpu_baseline.csv" -Encoding utf8
$monitor = $null
$runTimer = [Diagnostics.Stopwatch]::StartNew()
try {
    $monitor = Start-Process -FilePath $smi -ArgumentList @('--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu', '--format=csv,nounits', "--loop-ms=$($cfg.diagnostics.nvidia_poll_ms)") -WindowStyle Hidden -RedirectStandardOutput $csv -RedirectStandardError "$root\logs\m1_gpu_$stamp.stderr.log" -PassThru
    $matlabCommand = "addpath('$PSScriptRoot'); m1_single_tx('$ConfigPath','$outdir');"
    & $cfg.paths.matlab_exe -sd $root -logfile $log -batch $matlabCommand
    $matlabExit = $LASTEXITCODE
    if ($matlabExit -ne 0) { throw "MATLAB failed with exit $matlabExit. No retry or software installation. See $log" }
    if (-not (Test-Path -LiteralPath "$outdir\metrics.json")) { throw 'No success metrics produced' }
} finally {
    $runTimer.Stop()
    if ($monitor -and -not $monitor.HasExited) { Stop-Process -Id $monitor.Id }
    [pscustomobject]@{process_wall_s=$runTimer.Elapsed.TotalSeconds;output=$outdir;matlab_log=$log;gpu_csv=$csv} | ConvertTo-Json | Set-Content -LiteralPath "$outdir\runner.json" -Encoding utf8
}
$metrics = Get-Content -LiteralPath "$outdir\metrics.json" -Raw | ConvertFrom-Json
$samples = @(Import-Csv -LiteralPath $csv | Where-Object { $_.timestamp })
$start = [datetime]::ParseExact($metrics.solver_started_local, 'yyyy-MM-dd HH:mm:ss.fff', [Globalization.CultureInfo]::InvariantCulture)
$end = [datetime]::ParseExact($metrics.solver_finished_local, 'yyyy-MM-dd HH:mm:ss.fff', [Globalization.CultureInfo]::InvariantCulture)
$during = @($samples | Where-Object { $at=[datetime]::ParseExact($_.timestamp.Trim(),'yyyy/MM/dd HH:mm:ss.fff',[Globalization.CultureInfo]::InvariantCulture); $at -ge $start -and $at -le $end })
$usedName = ($samples[0].PSObject.Properties.Name | Where-Object { $_ -like '*memory.used*' })
$utilName = ($samples[0].PSObject.Properties.Name | Where-Object { $_ -like '*utilization.gpu*' })
$used = @($during | ForEach-Object { [double]$_.$usedName })
$util = @($during | ForEach-Object { [double]$_.$utilName })
$payload = ($cfg.array.elements*$cfg.array.elements*$cfg.time.output_count + ($cfg.model.label_shape[0]*$cfg.model.label_shape[1]))*4
$summary = [pscustomobject]@{
    solver_seconds = $metrics.solver_wall_s
    monitor_scope = 'Whole GPU, includes desktop and other processes; sampled every 500 ms; not process-specific peak allocation.'
    solver_gpu_sample_count = $during.Count
    solver_gpu_peak_used_MiB = ($used | Measure-Object -Maximum).Maximum
    solver_gpu_mean_utilization_percent = ($util | Measure-Object -Average).Average
    fmc_sample_seconds_linear = $cfg.array.elements*$metrics.solver_wall_s
    fmc_sample_seconds_with_margin = $cfg.array.elements*$metrics.solver_wall_s*(1+$cfg.diagnostics.estimate_overhead_fraction)
    samples_300_hours_linear = 300*$cfg.array.elements*$metrics.solver_wall_s/3600
    samples_500_hours_linear = 500*$cfg.array.elements*$metrics.solver_wall_s/3600
    fmc_and_label_bytes_per_sample = $payload
    samples_300_GiB_payload = 300*$payload/1GB
    samples_500_GiB_payload = 500*$payload/1GB
    free_E_GiB = (Get-PSDrive E).Free/1GB
}
$summary | ConvertTo-Json | Set-Content -LiteralPath "$outdir\summary.json" -Encoding utf8
$summary | ConvertTo-Json
Write-Output "M1_OUTPUT=$outdir"
