param(
    # 放 models/ 與 engines/mortal/Mortal/ 的位置。預設是這個專案目錄。
    [string]$AssetSource = "",
    # 離線 Chromium 的位置。不給就用建置環境自己的 Playwright 下載到 out/ms-playwright,
    # 版本一定對得上包進去的 Playwright。
    [string]$BrowserRoot = "",
    [string]$PublicKey = ""
)
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $Project
if (-not $AssetSource) { $AssetSource = $Project }
$env:YOLO_AUTOINSTALL = "false"

function RunChecked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe" }
}
# Build environments belong to this checkout, not an existing user's environment.
RunChecked "py" @("-3.14", "-m", "venv", "out/build-gui")
RunChecked "py" @("-3.12", "-m", "venv", "out/build-mortal")
RunChecked "py" @("-3.12", "-m", "venv", "out/build-yolo")
$Gui = Join-Path $Project "out/build-gui/Scripts/python.exe"
$Mortal = Join-Path $Project "out/build-mortal/Scripts/python.exe"
$Yolo = Join-Path $Project "out/build-yolo/Scripts/python.exe"
RunChecked $Gui @("-m", "pip", "install", "-r", "requirements.txt", "-r", "packaging/requirements-build.txt")
if (-not $BrowserRoot) {
    $BrowserRoot = Join-Path $Project "out/ms-playwright"
    $env:PLAYWRIGHT_BROWSERS_PATH = $BrowserRoot
    RunChecked $Gui @("-m", "playwright", "install", "chromium")
    Remove-Item Env:PLAYWRIGHT_BROWSERS_PATH
}
RunChecked $Mortal @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
RunChecked $Mortal @("-m", "pip", "install", "numpy")
RunChecked $Yolo @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cpu")
RunChecked $Yolo @("-m", "pip", "install", "-r", "engines/yolo/requirements.txt")
$BuildArgs = @("tools/package.py", "--asset-source", $AssetSource,
    "--build-tools", (Join-Path $Project "out/build-gui/Lib/site-packages"),
    "--main-python", $Gui, "--mortal-python", $Mortal, "--yolo-python", $Yolo,
    "--browser-root", $BrowserRoot)
if ($PublicKey) { $BuildArgs += @("--public-key", $PublicKey) }
RunChecked $Gui $BuildArgs
Write-Host "Next: compile packaging/windows/mia.iss with Inno Setup. No release was published."
