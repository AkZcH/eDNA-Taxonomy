# Setup script for Vite+React conversion
$frontendDir = "c:\Users\KIIT0001\Downloads\sih25\eDNA-Taxonomy\frontend"
Set-Location $frontendDir

Write-Host "Setting up Vite+React project..." -ForegroundColor Green

# Run the conversion script
if (Test-Path "$frontendDir\convert-jsx.ps1") {
    Write-Host "Converting JS components to JSX..." -ForegroundColor Cyan
    & "$frontendDir\convert-jsx.ps1"
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
npm install

Write-Host "Setup completed!" -ForegroundColor Green
Write-Host "To start the development server, run: npm run dev" -ForegroundColor Yellow