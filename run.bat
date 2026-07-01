cd test_module

taskkill /F /IM python.exe
rmdir /s /q build
del test_module*.pyd

python setup.py build_ext --inplace

if exist test_module*.pyd (
    powershell Write-Host "[SUCCESS] .pyd exists in gui_tools\pathology_detector_cpp" -ForegroundColor Green
    cd ..
    python tests.py
) else (
    powershell Write-Host "[WARNING] No .pyd file found in the local folder" -ForegroundColor Red
)
