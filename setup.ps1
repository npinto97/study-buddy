# Study Buddy - Automated Setup Script
# This script automates the installation process for the Study Buddy application

Write-Host "=== Study Buddy Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.1[0-9]") {
    Write-Host "✓ Python version OK: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.10+ required. Current: $pythonVersion" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Detect GPU
Write-Host ""
Write-Host "Detecting GPU..." -ForegroundColor Yellow
$hasNvidiaGPU = $false
try {
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasNvidiaGPU = $true
        Write-Host "✓ NVIDIA GPU detected: $gpuInfo" -ForegroundColor Green
    }
} catch {
    Write-Host "No NVIDIA GPU detected or nvidia-smi not available" -ForegroundColor Gray
}

# Install PyTorch
Write-Host ""
if ($hasNvidiaGPU) {
    Write-Host "Installing PyTorch with CUDA 12.8 support..." -ForegroundColor Yellow
    Write-Host "This will download ~2.8 GB. Please wait..." -ForegroundColor Gray
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Installing PyTorch (CPU-only version)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Install other dependencies
Write-Host ""
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Dependency installation failed" -ForegroundColor Red
    exit 1
}

# Install additional packages that might be missing
Write-Host ""
Write-Host "Installing additional packages..." -ForegroundColor Yellow
pip install ffmpeg-python youtube-transcript-api wikipedia google-search-results

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run:" -ForegroundColor Cyan
Write-Host "  streamlit run streamlit_frontend.py" -ForegroundColor White
Write-Host ""
Write-Host "Make sure you have configured your .env file with the required API keys." -ForegroundColor Yellow
