param(
 [string]$ConfigPath='E:\FNO_research\repos\fno_light\repro_lu2026\configs\kwave_reference.json',
 [int]$Processes=1,
 [int[]]$Transmitters=(1..64),
 [string]$RunName='reference'
)
$ErrorActionPreference='Stop'
$ConfigPath=(Resolve-Path -LiteralPath $ConfigPath).Path
$cfg=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$root=$cfg.paths.root
if($root -ne 'E:\FNO_research'){throw 'Unexpected root'}
$env:TEMP="$root\tmp"; $env:TMP=$env:TEMP
$env:CUDA_CACHE_PATH="$root\cache\cuda"
$env:PIP_CACHE_DIR="$root\cache\pip"; $env:CONDA_PKGS_DIRS="$root\cache\conda"
$env:TORCH_HOME="$root\cache\torch"; $env:MPLCONFIGDIR="$root\cache\matplotlib"
$env:PYTHONDONTWRITEBYTECODE='1';$env:XDG_CACHE_HOME="$root\cache"
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$dir="$root\results\${RunName}_${stamp}"
if(Test-Path $dir){throw 'Refusing existing run directory'}
New-Item -ItemType Directory -Path $dir | Out-Null
Copy-Item -LiteralPath $ConfigPath -Destination "$dir\config.json"
$logdir="$root\logs\${RunName}_${stamp}";New-Item -ItemType Directory -Path $logdir | Out-Null
$smi=(Get-Command nvidia-smi.exe).Source
& $smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,nounits | Set-Content "$dir\gpu_baseline.csv" -Encoding utf8
$jobs=@();$monitor=$null;$clock=[Diagnostics.Stopwatch]::StartNew()
$exitOkay=$false
try {
 $monitor=Start-Process -FilePath $smi -ArgumentList @('--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu','--format=csv,nounits',"--loop-ms=$($cfg.diagnostics.nvidia_poll_ms)") -WindowStyle Hidden -RedirectStandardOutput "$logdir\gpu.csv" -RedirectStandardError "$logdir\gpu.stderr.log" -PassThru
 for($i=0;$i -lt $Processes;$i++){
  $worker="$dir\worker_$($i+1)";New-Item -ItemType Directory -Path $worker | Out-Null
  foreach($sub in @('prefs','appdata','localappdata')){New-Item -ItemType Directory -Path "$worker\$sub" | Out-Null}
  $env:MATLAB_PREFDIR="$worker\prefs";$env:APPDATA="$worker\appdata";$env:LOCALAPPDATA="$worker\localappdata"
  $tx=@(for($j=$i;$j -lt $Transmitters.Count;$j+=$Processes){$Transmitters[$j]})
  $txLiteral=$tx -join ' '
  $matlab="addpath('$($cfg.paths.code_root)'); kwave_fmc_worker('$ConfigPath','$worker',[$txLiteral],'$dir\go.flag');"
  $args="-wait -sd `"$root`" -logfile `"$logdir\worker_$($i+1).log`" -batch `"$matlab`""
  $proc=Start-Process -FilePath $cfg.paths.matlab_exe -ArgumentList $args -WindowStyle Hidden -PassThru
  $jobs+= [pscustomobject]@{Process=$proc;Directory=$worker;Transmitters=$tx}
 }
 $readyClock=[Diagnostics.Stopwatch]::StartNew()
 while($true){
  $readyCount=0
  foreach($j in $jobs){
   if(Test-Path "$($j.Directory)\failure.txt"){throw "Worker failed: $($j.Directory)"}
   if(Test-Path "$($j.Directory)\ready.json"){$readyCount++}
   elseif($j.Process.HasExited){throw 'Worker exited before ready'}
  }
  if($readyCount -eq $Processes){break}
  if($readyClock.Elapsed.TotalSeconds -gt $cfg.benchmark.barrier_timeout_s){throw 'Readiness timeout'}
  Start-Sleep -Milliseconds $cfg.benchmark.poll_ms
 }
 $startup=$clock.Elapsed.TotalSeconds
 [IO.File]::WriteAllText("$dir\go.flag",'go')
 Write-Output "RUN_DIR=$dir PROCESSES=$Processes SHOTS=$($Transmitters.Count) STARTUP_S=$startup"
 while($true){
  foreach($j in $jobs){if(Test-Path "$($j.Directory)\failure.txt"){throw "Worker failed: $($j.Directory)"}}
  if(@($jobs|Where-Object {-not $_.Process.HasExited}).Count -eq 0){break}
  Start-Sleep -Milliseconds $cfg.benchmark.poll_ms
 }
 foreach($j in $jobs){if(-not(Test-Path "$($j.Directory)\metrics.json")){throw 'Worker has no success metrics'}}
 $exitOkay=$true
} finally {
 $clock.Stop()
 if($monitor -and -not $monitor.HasExited){Stop-Process -Id $monitor.Id}
 if(-not $exitOkay){
  foreach($j in $jobs){
   $pidFile="$($j.Directory)\process.json"
   if(Test-Path $pidFile){$ownPid=(Get-Content $pidFile -Raw|ConvertFrom-Json).pid;$p=Get-Process -Id $ownPid -ErrorAction SilentlyContinue;if($p -and $p.Path -like 'D:\matlab\*'){Stop-Process -Id $ownPid}}
   if(-not $j.Process.HasExited){Stop-Process -Id $j.Process.Id -ErrorAction SilentlyContinue}
  }
 }
 [pscustomobject]@{success=$exitOkay;processes=$Processes;transmitters=$Transmitters;process_wall_s=$clock.Elapsed.TotalSeconds;output=$dir;logdir=$logdir} | ConvertTo-Json -Depth 5 | Set-Content "$dir\runner.json" -Encoding utf8
}
$metas=@($jobs|ForEach-Object {Get-Content "$($_.Directory)\metrics.json" -Raw|ConvertFrom-Json})
$fmt='yyyy-MM-dd HH:mm:ss.fff';$culture=[Globalization.CultureInfo]::InvariantCulture
$starts=@($metas|ForEach-Object {[datetime]::ParseExact($_.started_local,$fmt,$culture)})
$ends=@($metas|ForEach-Object {[datetime]::ParseExact($_.finished_local,$fmt,$culture)})
$begin=($starts|Sort-Object|Select-Object -First 1);$finish=($ends|Sort-Object|Select-Object -Last 1)
$elapsed=($finish-$begin).TotalSeconds
$samples=@(Import-Csv "$logdir\gpu.csv"|Where-Object {$at=[datetime]::ParseExact($_.timestamp.Trim(),'yyyy/MM/dd HH:mm:ss.fff',$culture);$at -ge $begin -and $at -le $finish})
$usedName=$samples[0].PSObject.Properties.Name|Where-Object {$_ -like '*memory.used*'}
$utilName=$samples[0].PSObject.Properties.Name|Where-Object {$_ -like '*utilization.gpu*'}
$summary=[pscustomobject]@{processes=$Processes;shots=$Transmitters.Count;startup_s=$startup;compute_makespan_s=$elapsed;process_wall_s=$clock.Elapsed.TotalSeconds;shots_per_s=$Transmitters.Count/$elapsed;seconds_per_shot=$elapsed/$Transmitters.Count;gpu_peak_whole_card_MiB=(@($samples|ForEach-Object {[double]$_.$usedName})|Measure-Object -Maximum).Maximum;gpu_mean_utilization_percent=(@($samples|ForEach-Object {[double]$_.$utilName})|Measure-Object -Average).Average;gpu_samples=$samples.Count;output=$dir}
$summary|ConvertTo-Json|Set-Content "$dir\summary.json" -Encoding utf8
$summary|ConvertTo-Json
