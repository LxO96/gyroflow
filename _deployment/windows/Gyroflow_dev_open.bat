@echo off
rem Launches a DEVELOPMENT build of Gyroflow to open a file, for use as a Windows file
rem association ("Open with" -> this script).
rem
rem Two things make a dev build unsuitable for associating directly with gyroflow.exe:
rem
rem  1. target\release\gyroflow.exe does not have Qt, OpenCV or ffmpeg beside it - `just run`
rem     supplies them through PATH, and Explorer does not. Launching it from Explorer fails
rem     with "the code execution cannot proceed because <something>.dll was not found".
rem     A deployed build (`just deploy`) has those DLLs copied next to the exe and needs none
rem     of this; this script exists for the dev tree.
rem
rem  2. Gyroflow treats a bare file argument as a RENDER request (see cli.rs
rem     `will_run_in_console`). Opening a file in the GUI requires --open, so double-clicking
rem     a video associated straight to the exe would start a headless render instead.
rem
rem Usage: associate a file type with this script; it receives the path as %1.

setlocal

rem Repo root, derived from this script's own location (_deployment\windows\..\..).
set "ROOT=%~dp0..\.."

for /d %%D in ("%ROOT%\ext\6.7.3\msvc*") do set "QTBIN=%%D\bin"
rem Keep this in step with FFmpeg in _scripts/common.just - an old ext\ffmpeg-* directory is
rem left behind by an upgrade, so this names the version rather than globbing for one.
set "FFBIN=%ROOT%\ext\ffmpeg-9.0-windows-desktop-vs2026-gpl-lite\bin"
set "CVBIN=%ROOT%\ext\vcpkg\installed\x64-windows\bin"

set "PATH=%QTBIN%;%FFBIN%;%CVBIN%;%PATH%"

if "%~1"=="" (
    start "" "%ROOT%\target\release\gyroflow.exe"
) else (
    start "" "%ROOT%\target\release\gyroflow.exe" --open "%~1"
)

endlocal
