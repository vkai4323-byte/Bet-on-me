$ErrorActionPreference = "Stop"
$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $Python -m sportsbet_tool.cli @args
