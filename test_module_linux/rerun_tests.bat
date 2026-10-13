:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

@echo off
SETLOCAL Enabledelayedexpansion
chcp 65001 >nul


echo "Run install.bat before"
call "%~dp0..\print_box.bat" "Running Benchmark Suite Inside Pure Linux Environment..."
REM Check that Docker CLI is installed
echo Is Docker CLI installed...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH.
    exit /b 1
)

REM Check that Docker Engine is running
echo Is Docker Engine running...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop and try again.
    exit /b 1
)

REM Check that the Compose service is running
echo Is Compose service running...
docker compose ps --status running --services | findstr /x /c:"hpc_stack" >nul
if errorlevel 1 (
    echo ERROR: Container hpc_stack is not running.
    echo Starting containers...
    docker compose up -d
    if errorlevel 1 exit /b 1
)

REM Run tests inside the container
echo Running tests.py...
docker compose exec -T hpc_stack bash -c "python3 -u tests.py"
set "rc=%ERRORLEVEL%"

if not "%rc%"=="0" (
    echo ERROR: Tests failed with exit code %rc%.
) else (
    echo Tests passed.
)

call "%~dp0..\print_box.bat" "Test Pipeline Execution Finished!"
pause



@REM :: usefull commands:
@REM :: 1. Shut down the current container
@REM docker compose down
@REM
@REM Update docker image
@REM :: 2. Force a completely clean build ignoring cached layers
@REM @REM docker compose build --no-cache
@REM docker compose build
@REM
@REM :: 3. Boot it back up
@REM docker compose up -d
@REM
@REM :: open docker container terminal
@REM docker compose exec hpc_stack bash

@REM docker exec -it hpc_test_runner bash
@REM > nvidia-smi
@REM > nvfortran --version
@REM > echo "$CUDA_VISIBLE_DEVICES"


@REM docker compose up -d hpc_stack
@REM docker compose ps -a
@REM docker compose exec hpc_stack bash -lc "cd /workspace/test_module_linux && python3 -c 'import hpc_benchmark_core; print(hpc_benchmark_core.__file__)'"
