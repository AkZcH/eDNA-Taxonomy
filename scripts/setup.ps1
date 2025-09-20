# PowerShell script to set up the development environment

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Create necessary directories
Write-Host "Creating project directories..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path ..\uploads | Out-Null
New-Item -ItemType Directory -Force -Path ..\tests\data | Out-Null

# Check Python installation
if (-not (Test-Command python)) {
    Write-Host "Error: Python is not installed. Please install Python 3.8 or later." -ForegroundColor Red
    exit 1
}

# Check pip installation
if (-not (python -m pip --version)) {
    Write-Host "Error: pip is not accessible via 'python -m pip'. Please ensure Python and pip are correctly installed." -ForegroundColor Red
    exit 1
}

# Check Node.js installation
if (-not (Test-Command node)) {
    Write-Host "Error: Node.js is not installed. Please install Node.js 14 or later." -ForegroundColor Red
    exit 1
}

# Check npm installation
if (-not (Test-Command npm)) {
    Write-Host "Error: npm is not installed. Please install npm." -ForegroundColor Red
    exit 1
}

# Create and activate virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Green
python -m venv ..\venv
..\venv\Scripts\Activate

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r ..\requirements.txt

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Green
Set-Location ..\frontend
npm install
Set-Location ..\scripts

# Create .env file if it doesn't exist
if (-not (Test-Path ..\'.env')) {
    Write-Host "Creating .env file..." -ForegroundColor Green
    @"
FLASK_ENV=development
MONGODB_URI=mongodb+srv://ritamvaskar0:Ritam2005@cluster0.lklwwgr.mongodb.net//edna_taxonomy
MODEL_PATH=multitask_model_demo_small.pth
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
"@ | Out-File -FilePath ..\'.env'
}

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Green
python .\init_db.py

Write-Host "
Setup completed successfully!" -ForegroundColor Green
Write-Host "
To start the application:
1. Activate the virtual environment: .\venv\Scripts\Activate
2. Start the backend: python app.py
3. In a new terminal, start the frontend: cd frontend && npm start
" -ForegroundColor Yellow