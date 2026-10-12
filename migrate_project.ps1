$ErrorActionPreference='Stop'
$taskSourceRoot='C:\Users\Administrator\Desktop\大二下\大三课程\文献阅读'
$taskDestinationRoot='D:\下载\FNO_Research_Workspace'
$taskSourceResolved=(Resolve-Path -LiteralPath $taskSourceRoot).Path.TrimEnd('\')
$taskDestinationResolved=(Resolve-Path -LiteralPath $taskDestinationRoot).Path.TrimEnd('\')
if(-not $taskSourceResolved.StartsWith('C:\Users\Administrator\Desktop\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected source root'}
if(-not $taskDestinationResolved.StartsWith('D:\下载\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected destination root'}
$taskManifest=New-Object System.Collections.Generic.List[object]
foreach($taskName in @('.build','Research_Direction','PINO超声成像知识库')){
    $taskSource=Join-Path $taskSourceResolved $taskName
    $taskDestination=Join-Path $taskDestinationResolved $taskName
    $taskBackup=Join-Path $taskSourceResolved ($taskName+'.migration-backup-20261009')
    if(Test-Path -LiteralPath $taskDestination){throw "Destination exists: $taskDestination"}
    if(Test-Path -LiteralPath $taskBackup){throw "Backup exists: $taskBackup"}
    $taskResolved=(Resolve-Path -LiteralPath $taskSource).Path
    if(-not $taskResolved.StartsWith($taskSourceResolved+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Source escaped workspace'}
    if((Get-Item -LiteralPath $taskSource -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Source already reparse point'}
    if(@(Get-ChildItem -LiteralPath $taskSource -Recurse -Force -Attributes ReparsePoint).Count){throw 'Nested reparse point requires review'}
    Write-Output "Copying $taskName"
    $taskLog=Join-Path $taskDestinationResolved ('migration-'+$taskName+'.log')
    & robocopy $taskSource $taskDestination /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NP "/LOG:$taskLog" | Out-Null
    if($LASTEXITCODE -ge 8){throw "Robocopy failed: $LASTEXITCODE"}
    $taskFiles=@(Get-ChildItem -LiteralPath $taskSource -Recurse -Force -File)
    $taskDestFiles=@(Get-ChildItem -LiteralPath $taskDestination -Recurse -Force -File)
    if($taskFiles.Count -ne $taskDestFiles.Count){throw 'File counts differ'}
    $taskIndex=0
    foreach($taskFile in $taskFiles){
        $taskRelative=$taskFile.FullName.Substring($taskSource.Length).TrimStart('\')
        $taskDestFile=Join-Path $taskDestination $taskRelative
        if((Get-Item -LiteralPath $taskDestFile).Length -ne $taskFile.Length){throw "Size mismatch: $taskRelative"}
        $taskHash=(Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash
        if((Get-FileHash -LiteralPath $taskDestFile -Algorithm SHA256).Hash -ne $taskHash){throw "Hash mismatch: $taskRelative"}
        $taskManifest.Add([pscustomobject]@{Source=$taskFile.FullName;Destination=$taskDestFile;Bytes=$taskFile.Length;SHA256=$taskHash})
        $taskIndex++
        if(($taskIndex % 500) -eq 0){Write-Output "$taskName verified $taskIndex / $($taskFiles.Count)"}
    }
    $taskManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskDestinationResolved 'project-migration-manifest.json') -Encoding utf8
    Write-Output "Verified $taskName; switching to D-drive junction"
    Move-Item -LiteralPath $taskSource -Destination $taskBackup
    New-Item -ItemType Junction -Path $taskSource -Target $taskDestination | Out-Null
    $taskLink=Get-Item -LiteralPath $taskSource -Force
    if(-not ($taskLink.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Junction not created'}
    if($taskLink.Target -ne $taskDestination){throw 'Junction target differs'}
    # Remove the verified redundant copy only after checking the absolute literal backup path.
    $taskBackupResolved=(Resolve-Path -LiteralPath $taskBackup).Path
    if($taskBackupResolved -ne $taskBackup -or -not $taskBackupResolved.StartsWith($taskSourceResolved+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Unsafe backup path'}
    if((Get-Item -LiteralPath $taskBackup -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Backup unexpectedly linked'}
    Remove-Item -LiteralPath $taskBackupResolved -Recurse -Force
    Write-Output "Migrated ${taskName}: $($taskFiles.Count) files"
}
$taskSummary=[pscustomobject]@{Status='completed';Files=$taskManifest.Count;Bytes=($taskManifest | Measure-Object Bytes -Sum).Sum;Destination=$taskDestinationResolved;Compatibility='C paths are directory junctions; physical files on D';HashVerification='SHA256 of every source/destination pair before removing original';Completed=(Get-Date).ToString('o')}
$taskSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDestinationResolved 'project-migration-summary.json') -Encoding utf8
$taskSummary | ConvertTo-Json
