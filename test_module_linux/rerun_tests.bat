:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

@echo off
SETLOCAL Enabledelayedexpansion

:: 1. Navigate to the script's directory
F:
cd \usr\PycharmProjects\VEC_tests\test_module_linux


echo =======================================================================
echo Running Benchmark Suite Inside Pure Linux Environment...
echo =======================================================================
docker compose exec -T hpc_stack bash -c "python3 tests.py"

echo =======================================================================
echo Test Pipeline Execution Finished!
echo =======================================================================
pause



@REM :: usefull commands:
@REM :: 1. Shut down the current container
@REM docker compose down
@REM
@REM :: 2. Force a completely clean build ignoring cached layers
@REM docker compose build --no-cache
@REM
@REM :: 3. Boot it back up
@REM docker compose up -d
@REM
@REM :: open docker container terminal
@REM docker compose exec hpc_stack bash