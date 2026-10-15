$ErrorActionPreference='Stop'
$demoRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$backend=(Resolve-Path -LiteralPath (Join-Path $demoRoot 'backend\app.py')).Path
$pidFile=Join-Path $demoRoot 'data\server.pid'
if(-not(Test-Path -LiteralPath $pidFile)){Write-Output '이 실행 스크립트가 관리하는 서버 PID가 없습니다.';return}
$serverProcessId=[int]([IO.File]::ReadAllText($pidFile).Trim())
$process=Get-CimInstance Win32_Process -Filter "ProcessId=$serverProcessId" -ErrorAction SilentlyContinue
if($null -eq $process){Write-Output '서버가 이미 종료되었습니다.';return}
if($process.CommandLine -notlike ('*'+$backend+'*')){throw 'PID가 이 대시보드 서버와 일치하지 않아 종료하지 않습니다.'}
Stop-Process -Id $serverProcessId
Write-Output 'FactoryLens 서버를 종료했습니다. 원본과 저장 로그는 보존했습니다.'
