# PowerShell script to run development servers

# Function to check if a process is running on a port
function Test-Port($port) {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

# Function to start the Flask backend server
function Start-Backend {
    Write-Host "Starting Flask backend server..." -ForegroundColor Green
    Set-Location ..
    
    # Activate virtual environment
    .\venv\Scripts\Activate
    
    # Check if port 5000 is available
    if (Test-Port 5000) {
        Write-Host "Error: Port 5000 is already in use. Please free up the port and try again." -ForegroundColor Red
        return $false
    }
    
    # Start Flask server
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        Set-Location '$PWD';
        .\venv\Scripts\Activate;
        python app.py;
    "
    
    # Wait for server to start
    $attempts = 0
    while (-not (Test-Port 5000) -and $attempts -lt 10) {
        Start-Sleep -Seconds 1
        $attempts++
    }
    
    if (-not (Test-Port 5000)) {
        Write-Host "Error: Failed to start Flask server" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Function to start the Vite frontend server
function Start-Frontend {
    Write-Host "Starting Vite frontend server..." -ForegroundColor Green
    Set-Location ..\frontend
    
    # Check if port 3000 is available
    if (Test-Port 3000) {
        Write-Host "Error: Port 3000 is already in use. Please free up the port and try again." -ForegroundColor Red
        return $false
    }
    
    # Start Vite development server
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "
        Set-Location '$PWD';
        npm run dev;
    "
    
    # Wait for server to start
    $attempts = 0
    while (-not (Test-Port 3000) -and $attempts -lt 10) {
        Start-Sleep -Seconds 1
        $attempts++
    }
    
    if (-not (Test-Port 3000)) {
        Write-Host "Error: Failed to start React server" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Main execution
Write-Host "Starting development servers..." -ForegroundColor Cyan

# Start backend server
if (-not (Start-Backend)) {
    exit 1
}

# Start frontend server
if (-not (Start-Frontend)) {
    exit 1
}

Write-Host "
Development servers started successfully!" -ForegroundColor Green
Write-Host "
Backend server running at: http://localhost:5000
Frontend server running at: http://localhost:3000
" -ForegroundColor Yellow

Write-Host "Press Ctrl+C to stop the servers" -ForegroundColor Cyan

# Keep the script running
while ($true) {
    Start-Sleep -Seconds 1
}