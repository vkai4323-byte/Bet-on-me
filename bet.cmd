@echo off
set PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
"%PYTHON%" -m sportsbet_tool.cli %*
