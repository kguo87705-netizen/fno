param([Parameter(Mandatory=$true)][string]$ConfigPath)
$ErrorActionPreference='Stop'
$cfg=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$root=$cfg.root
$series=Join-Path "$root/results" ('adjudication_kwave_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $series | Out-Null
Copy-Item -LiteralPath $ConfigPath -Destination "$series/config.json"
$records=@()
foreach($tag in @('dt25','dt125','pml20')){
 foreach($medium in @('defect','uniform')){
  $name="adjudication_kwave_${tag}_${medium}"
  $config=Join-Path (Split-Path $ConfigPath) ($name+'.json')
  $lines=& "$PSScriptRoot/run_kwave_group.ps1" -ConfigPath $config -Processes 1 -Transmitters $cfg.transmitters_one_based -RunName $name
  $lines | Write-Output
  $runs=@(Get-ChildItem -LiteralPath "$root/results" -Directory -Filter "${name}_*" | Sort-Object Name)
  $dir=$runs[-1].FullName
  $summary=Get-Content "$dir/summary.json" -Raw | ConvertFrom-Json
  $records += [pscustomobject]@{variant=$tag;medium=$medium;directory=$dir;summary=$summary}
  $records | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 "$series/runs.json"
 }
}
Write-Output "SERIES_DIR=$series"
