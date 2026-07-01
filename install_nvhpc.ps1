Write-Host "--- Generating Clean Native Linux Installation Script ---" -ForegroundColor Cyan

# Create the shell script block
$linuxScript = @'
#!/bin/bash
set -e
echo "=== Starting Internal Installation Loop ==="

# 1. Update and install prerequisites
sudo apt-get update
sudo apt-get install -y wget gnupg2 curl

# 2. Add NVIDIA HPC SDK repository key safely
curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg

# 3. Add the precise repository metadata line
echo "deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" | sudo tee /etc/apt/sources.list.d/nvhpc.list

# 4. Update index and download the SDK packages
sudo apt-get update
sudo apt-get install -y nvhpc-24-5

# 5. Append environment variables safely to your .bashrc profile
sed -i '/hpc_sdk/d' ~/.bashrc
echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin:$PATH' >> ~/.bashrc
echo 'export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/man:$MANPATH' >> ~/.bashrc

echo "=== Installation Completed Successfully ==="
'@

# FIX: Force clean Linux Line Endings by completely stripping out Windows '\r' characters
$cleanLinuxScript = $linuxScript -replace "`r", ""

# Save it safely into the WSL file system
$cleanLinuxScript | wsl tee ~/install_nvhpc.sh > $null

Write-Host "--- Launching script inside WSL (Password required once) ---" -ForegroundColor Yellow
wsl chmod +x ~/install_nvhpc.sh
wsl bash ~/install_nvhpc.sh

Write-Host "--- Clean up installer file ---" -ForegroundColor Gray
wsl rm ~/install_nvhpc.sh

Write-Host "--- Done! Restart your WSL terminal or run 'source ~/.bashrc' ---" -ForegroundColor Green
