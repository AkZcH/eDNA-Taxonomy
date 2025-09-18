# Script to convert JS files to JSX
$frontendDir = "c:\Users\KIIT0001\Downloads\sih25\eDNA-Taxonomy\frontend"

# Create necessary directories
New-Item -ItemType Directory -Path "$frontendDir\src\assets" -Force
New-Item -ItemType Directory -Path "$frontendDir\public" -Force

# Files to convert
$files = @(
    @{Source="$frontendDir\src\components\ResultsViewer.js"; Target="$frontendDir\src\components\ResultsViewer.jsx"},
    @{Source="$frontendDir\src\components\TaxonomyViewer.js"; Target="$frontendDir\src\components\TaxonomyViewer.jsx"},
    @{Source="$frontendDir\src\pages\Dashboard.js"; Target="$frontendDir\src\pages\Dashboard.jsx"},
    @{Source="$frontendDir\src\pages\SampleUpload.js"; Target="$frontendDir\src\pages\SampleUpload.jsx"}
)

foreach ($file in $files) {
    if (Test-Path $file.Source) {
        $content = Get-Content -Path $file.Source -Raw
        # Update imports to use .jsx extension
        $content = $content -replace "from '([^']+)\.js'", "from '$1.jsx'"
        $content = $content -replace "import React from 'react';", "import React from 'react';"
        Set-Content -Path $file.Target -Value $content
        Write-Host "Converted $($file.Source) to $($file.Target)"
    } else {
        Write-Host "Source file $($file.Source) not found" -ForegroundColor Yellow
    }
}

Write-Host "Conversion completed successfully!" -ForegroundColor Green
Write-Host "Run 'npm install' and then 'npm run dev' to start the Vite development server"