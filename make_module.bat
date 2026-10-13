:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

cd test_module

REM Tente de tuer python si un processus bloque les fichiers
taskkill /F /IM python.exe 2>nul

REM Nettoyage des dossiers de build
if exist build rmdir /s /q build
if exist hpc_benchmark_core*.pyd del /f /q hpc_benchmark_core*.pyd

REM Compilation en place
python setup.py build_ext --inplace

cd ..
if exist test_module/hpc_benchmark_core*.pyd (
    powershell Write-Host "[SUCCESS] .pyd exists in gui_tools\pathology_detector_cpp" -ForegroundColor Green
    python tests.py
) else (
    powershell Write-Host "[WARNING] No .pyd file found in the local folder" -ForegroundColor Red
)
