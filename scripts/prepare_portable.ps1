# prepare_portable.ps1
# Builds a self-contained portable folder for the Windows installer.
# Run from the repository root on Windows (or in GitHub Actions windows-latest).
#
# Sources (official / trusted only):
#   Python  : https://www.python.org/ftp/python/  (embeddable package)
#   FFmpeg  : https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
#   get-pip : https://bootstrap.pypa.io/get-pip.py

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Portable = Join-Path $Root "portable"
$Runtime  = Join-Path $Portable "runtime"
$PythonDir = Join-Path $Runtime "python"
$FfmpegDir = Join-Path $Runtime "ffmpeg"

Write-Host "=== Preparing portable Video Clipper ===" -ForegroundColor Cyan

# Clean previous
if (Test-Path $Portable) { Remove-Item -Recurse -Force $Portable }
New-Item -ItemType Directory -Path $PythonDir -Force | Out-Null
New-Item -ItemType Directory -Path $FfmpegDir -Force | Out-Null

# ---------------------------------------------------------------------------
# 1. Download official Python embeddable (64-bit)
# ---------------------------------------------------------------------------
$PyVersion = "3.12.8"
$PyZip = "python-$PyVersion-embed-amd64.zip"
$PyUrl = "https://www.python.org/ftp/python/$PyVersion/$PyZip"
$PyZipPath = Join-Path $env:TEMP $PyZip

Write-Host "Downloading Python $PyVersion embeddable..."
Invoke-WebRequest -Uri $PyUrl -OutFile $PyZipPath -UseBasicParsing
Expand-Archive -Path $PyZipPath -DestinationPath $PythonDir -Force

# Enable site-packages in the embeddable distribution
$Pth = Get-ChildItem $PythonDir -Filter "python*._pth" | Select-Object -First 1
if ($Pth) {
    $content = Get-Content $Pth.FullName
    $content = $content -replace "#import site", "import site"
    if ($content -notcontains "Lib\\site-packages") {
        $content += "Lib\\site-packages"
    }
    Set-Content -Path $Pth.FullName -Value $content
}

# ---------------------------------------------------------------------------
# 2. Install pip into the embeddable Python
# ---------------------------------------------------------------------------
$GetPip = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip -UseBasicParsing
& "$PythonDir\python.exe" $GetPip --no-warn-script-location

# ---------------------------------------------------------------------------
# 3. Install project dependencies into the portable Python
# ---------------------------------------------------------------------------
Write-Host "Installing Python packages..."
& "$PythonDir\python.exe" -m pip install --no-warn-script-location -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Download FFmpeg essentials (gyan.dev - linked from ffmpeg.org)
# ---------------------------------------------------------------------------
$FfZip = "ffmpeg-release-essentials.zip"
$FfUrl = "https://www.gyan.dev/ffmpeg/builds/$FfZip"
$FfZipPath = Join-Path $env:TEMP $FfZip

Write-Host "Downloading FFmpeg essentials..."
Invoke-WebRequest -Uri $FfUrl -OutFile $FfZipPath -UseBasicParsing

$FfExtract = Join-Path $env:TEMP "ffmpeg-extract"
if (Test-Path $FfExtract) { Remove-Item -Recurse -Force $FfExtract }
Expand-Archive -Path $FfZipPath -DestinationPath $FfExtract -Force

# The zip contains a folder like ffmpeg-8.x-essentials_build
$Inner = Get-ChildItem $FfExtract -Directory | Select-Object -First 1
Copy-Item -Path (Join-Path $Inner.FullName "*") -Destination $FfmpegDir -Recurse -Force

# ---------------------------------------------------------------------------
# 5. Copy application source
# ---------------------------------------------------------------------------
Write-Host "Copying application files..."
$CopyItems = @("app.py", "requirements.txt", "src", "scripts", "temp", "output")
foreach ($item in $CopyItems) {
    $src = Join-Path $Root $item
    $dst = Join-Path $Portable $item
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Recurse -Force
    }
}
# Ensure empty dirs exist
New-Item -ItemType Directory -Path (Join-Path $Portable "temp") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Portable "output") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $Portable "temp\.gitkeep") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $Portable "output\.gitkeep") -Force | Out-Null

# ---------------------------------------------------------------------------
# 6. Create the .bat launcher (used by the desktop shortcut)
# ---------------------------------------------------------------------------
$Bat = @"
@echo off
cd /d "%~dp0"
set FFMPEG_PATH=%~dp0runtime\ffmpeg\bin\ffmpeg.exe
set PATH=%~dp0runtime\python;%~dp0runtime\ffmpeg\bin;%PATH%
"%~dp0runtime\python\pythonw.exe" "%~dp0scripts\launcher.py"
if errorlevel 1 (
    "%~dp0runtime\python\python.exe" "%~dp0scripts\launcher.py"
    pause
)
"@
Set-Content -Path (Join-Path $Portable "VideoClipper.bat") -Value $Bat -Encoding ASCII

Write-Host "Portable tree ready at: $Portable" -ForegroundColor Green
Write-Host "You can now compile installer\VideoClipper.iss with Inno Setup."
