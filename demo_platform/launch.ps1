param([switch]$OpenBrowser,[string]$PythonPath)
$ErrorActionPreference='Stop'
$demoRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$projectRoot=Split-Path -Parent $demoRoot
$python=if($PythonPath){(Resolve-Path -LiteralPath $PythonPath).Path}else{Join-Path $projectRoot '.venv\Scripts\python.exe'}
$backend=Join-Path $demoRoot 'backend\app.py'
$serverUrl='http://127.0.0.1:8099'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw '저장소 루트의 .venv 환경을 준비하고 requirements.txt를 설치하세요. 다른 환경은 -PythonPath로 python.exe 경로를 지정할 수 있습니다.'}
try{$ready=Invoke-RestMethod -Uri "$serverUrl/api/health" -TimeoutSec 2}catch{$ready=$null}
if($ready.product -ne 'FactoryLens'){
    $logDir=Join-Path $demoRoot 'data\logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $process=Start-Process -FilePath $python -ArgumentList @(('"'+$backend+'"')) -WorkingDirectory (Split-Path -Parent $backend) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'server.stdout.log') -RedirectStandardError (Join-Path $logDir 'server.stderr.log') -PassThru
    [IO.File]::WriteAllText((Join-Path $demoRoot 'data\server.pid'),[string]$process.Id,[Text.UTF8Encoding]::new($false))
    for($attempt=0;$attempt -lt 50;$attempt++){
        Start-Sleep -Milliseconds 500
        try{$ready=Invoke-RestMethod -Uri "$serverUrl/api/health" -TimeoutSec 1;if($ready.product -eq 'FactoryLens'){break}}catch{}
        if($process.HasExited){throw '서버가 종료되었습니다. data/logs의 진단 로그를 확인하세요.'}
    }
    if($ready.product -ne 'FactoryLens'){throw '서버 준비가 지연되고 있습니다. data/logs의 진단 로그를 확인하세요.'}
}
Write-Output "FactoryLens: $serverUrl"
Write-Output '실제 수집 시연은 화면의 재생 버튼으로 시작합니다. 기록은 data/platform.sqlite에 남습니다.'
if($OpenBrowser){Start-Process $serverUrl}
