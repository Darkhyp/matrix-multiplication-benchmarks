:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

cd test_module

taskkill /F /IM python.exe
rmdir /s /q build
del test_module*.pyd

python setup.py build_ext --inplace

cd ..
if exist test_module/test_module*.pyd (
    powershell Write-Host "[SUCCESS] .pyd exists in gui_tools\pathology_detector_cpp" -ForegroundColor Green
    python tests.py
) else (
    powershell Write-Host "[WARNING] No .pyd file found in the local folder" -ForegroundColor Red
)
