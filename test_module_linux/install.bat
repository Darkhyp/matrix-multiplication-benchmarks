:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

@echo off
SETLOCAL Enabledelayedexpansion
chcp 65001 >nul

call "%~dp0..\print_box.bat" "Checking Docker Daemon State..."
:: Check if Docker engine is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is NOT running!
    echo Please open your Windows Start Menu, launch Docker Desktop,
    echo wait for the whale icon to turn green, then run this script again.
    pause
    exit /b 1
)

echo Docker is running.

call "%~dp0..\print_box.bat" "Ensuring Container is up and Volumes are Remounted..."
:: '--build' checks if your Dockerfile changed. If it didn't, it skips
:: straight to starting the container using your live local folder changes.
docker compose up -d --build

call "%~dp0..\print_box.bat" "Running Benchmark Suite Inside Pure Linux Environment..."
:: This non-interactive command sends the script directly into the container,
:: runs it, streams the live output to your terminal, and exits back to CMD when done.
docker compose exec -T hpc_stack bash -c "chmod +x test_module_linux/make_module.sh"
docker compose exec -T hpc_stack bash -c "./test_module_linux/make_module.sh"

call "%~dp0..\print_box.bat" "Test Pipeline Execution Finished!"
pause
