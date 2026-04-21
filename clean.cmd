@echo off
title Clean

set /p DELETE_UV_CACHE="Do you want to clear global uv cache? (y/N) "
if /i "%DELETE_UV_CACHE%"=="y" (
    "tools\astral\uv.exe" cache clean >nul 2>&1
)

set /p DELETE_HF_CACHE="Do you want to remove Hugging Face cache? (y/N) "
if /i "%DELETE_HF_CACHE%"=="y" (
    rmdir /s /q "%USERPROFILE%\.cache\huggingface" >nul 2>&1
)

set /p DELETE_TRITON_CACHE="Do you want to delete Triton cache? (y/N) "
if /i "%DELETE_TRITON_CACHE%"=="y" (
    rmdir /s /q "%USERPROFILE%\.triton" >nul 2>&1
)

set /p DELETE_PROMPTS_DB="Do you want to drop prompts history DB? (y/N) "
if /i "%DELETE_PROMPTS_DB%"=="y" (
    del /q "%USERPROFILE%\.zpix\prompts_history.sqlite" >nul 2>&1
)
rem Prompts history as a SQLite DB was only available in ZPix v1.0.5.

echo Cleaning now local cache and temp files...
rmdir /s /q ".ruff_cache" >nul 2>&1
rmdir /s /q ".venv" >nul 2>&1
rmdir /s /q "build" >nul 2>&1
rmdir /s /q "source\py\__pycache__" >nul 2>&1
rmdir /s /q "temp" >nul 2>&1
rmdir /s /q "vcpkg_installed" >nul 2>&1

echo Done.
pause