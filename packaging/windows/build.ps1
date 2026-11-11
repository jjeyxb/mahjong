param(
    # 放 models/ 與 engines/mortal/Mortal/ 的位置。預設是這個專案目錄。
    [string]$AssetSource = "",
    # 離線 Chromium 的位置。不給就用建置環境自己的 Playwright 下載到 out/ms-playwright,
    # 版本一定對得上包進去的 Playwright。
    [string]$BrowserRoot = "",
    [string]$PublicKey = "",
    # 建 Mortal / YOLO 環境用的 Python 3.12。預設用 py 啟動器的 3.12。
    # ⚠ 不能是 3.12.0:它的 code.replace() 會把行內推導式改壞,而 PyInstaller 打包時對每個模組
    #   都做這件事 —— torch 的 _numpy/_ufuncs.py 打包後一載入就 NameError(2026-11-12 實測)。
    [string]$Python312 = ""
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
function NewVenv([string[]]$Base, [string]$Path) {
    # 環境已經在、但是用別的 Python 建的(例如換掉了有 bug 的 3.12.0),整個重建;
    # 只跑 venv 不加 --clear 的話,舊環境裡的直譯器不會換。
    $Exe = $Base[0]; $Pre = @($Base | Select-Object -Skip 1)
    $want = (& $Exe @Pre -c "import sys; print(sys.version.split()[0])").Trim()
    $cfg = Join-Path $Path "pyvenv.cfg"
    $have = if (Test-Path $cfg) { ((Get-Content $cfg | Where-Object { $_ -like "version*" }) -split "=")[-1].Trim() } else { "" }
    $Arguments = @($Pre + @("-m", "venv"))
    if ($have -and $have -ne $want) { $Arguments += "--clear" }
    RunChecked $Exe ($Arguments + @($Path))
}
# 宣告成字串陣列:PowerShell 會把 if 回傳的單一元素陣列展開成字串,$Base312[0] 就變成路徑的第一個字母。
[string[]]$Base312 = if ($Python312) { $Python312 } else { "py", "-3.12" }
$Pre312 = @($Base312 | Select-Object -Skip 1)
$Version312 = (& $Base312[0] @Pre312 -c "import sys; print(sys.version.split()[0])").Trim()
if ($Version312 -eq "3.12.0") {
    throw "Python 3.12.0 打包出來的 YOLO / Mortal 會壞(見 -Python312 的說明)。請改用 3.12.1 以上,用 -Python312 指定。"
}
# Build environments belong to this checkout, not an existing user's environment.
NewVenv @("py", "-3.14") "out/build-gui"
NewVenv $Base312 "out/build-mortal"
NewVenv $Base312 "out/build-yolo"
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
# PyInstaller 裝進兩個 3.12 環境自己:借主程式 3.14 環境的會連帶看到 3.14 編的套件,見 tools/package.py。
RunChecked $Mortal @("-m", "pip", "install", "-r", "packaging/requirements-build.txt")
RunChecked $Yolo @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cpu")
RunChecked $Yolo @("-m", "pip", "install", "-r", "engines/yolo/requirements.txt")
RunChecked $Yolo @("-m", "pip", "install", "-r", "packaging/requirements-build.txt")
$BuildArgs = @("tools/package.py", "--asset-source", $AssetSource,
    "--build-tools", (Join-Path $Project "out/build-gui/Lib/site-packages"),
    "--main-python", $Gui, "--mortal-python", $Mortal, "--yolo-python", $Yolo,
    "--browser-root", $BrowserRoot)
if ($PublicKey) { $BuildArgs += @("--public-key", $PublicKey) }
RunChecked $Gui $BuildArgs
Write-Host "Next: compile packaging/windows/mia.iss with Inno Setup. No release was published."
