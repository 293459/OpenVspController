[CmdletBinding()]
param(
    [ValidateSet("notebook", "setup", "verify", "streamlit")]
    [string]$Action = "notebook",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoInterpreterDir = Join-Path $RepoRoot "interpreter"
$RepoInterpreterPython = Join-Path $RepoInterpreterDir "python.exe"
$RepoLibsDir = Join-Path $RepoRoot "libs"
$RepoTempDir = Join-Path $RepoRoot ".tmp"
$RepoPipCacheDir = Join-Path $RepoTempDir "pip-cache"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$NotebookPath = Join-Path $RepoRoot "notebooks\main_analysis.ipynb"
$StreamlitApp = Join-Path $RepoRoot "app\streamlit_app.py"
$VerifyScript = Join-Path $RepoRoot "scripts\verify_setup.py"
$BootstrapPipScript = Join-Path $RepoRoot "scripts\bootstrap_pip.py"
$BootstrapRuntimeScript = Join-Path $RepoRoot "scripts\bootstrap_python_runtime.py"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"
$KernelName = "openvsp-controller"
$KernelDisplayName = "OpenVSP Controller (.venv)"
$RequiredPythonVersion = "3.13"
$RepoPathBootstrapFileName = "openvsp_controller_repo.pth"

function Write-Step {
    param([string]$Message)
    Write-Host "[OpenVSP Controller] $Message"
}

function Invoke-External {
    param(
        [string[]]$CommandParts,
        [string]$Description,
        [string]$WorkingDirectory = $RepoRoot,
        [hashtable]$EnvironmentOverrides = @{}
    )

    Write-Step $Description
    Push-Location $WorkingDirectory
    $originalEnvironment = @{}
    try {
        foreach ($key in $EnvironmentOverrides.Keys) {
            $originalEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
            [Environment]::SetEnvironmentVariable($key, [string]$EnvironmentOverrides[$key], "Process")
        }

        $command = $CommandParts[0]
        $arguments = @()
        if ($CommandParts.Length -gt 1) {
            $arguments = $CommandParts[1..($CommandParts.Length - 1)]
        }

        & $command @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $($CommandParts -join ' ')"
        }
    }
    finally {
        foreach ($key in $EnvironmentOverrides.Keys) {
            [Environment]::SetEnvironmentVariable($key, $originalEnvironment[$key], "Process")
        }
        Pop-Location
    }
}

function Invoke-PythonCapture {
    param(
        [string[]]$CommandParts,
        [string]$Code
    )

    try {
        $command = $CommandParts[0]
        $arguments = @()
        if ($CommandParts.Length -gt 1) {
            $arguments = $CommandParts[1..($CommandParts.Length - 1)]
        }

        $output = & $command @arguments -c $Code 2>$null
        if ($LASTEXITCODE -eq 0) {
            return (($output | Out-String).Trim())
        }
    }
    catch {
        return ""
    }

    return ""
}

function Test-CompatiblePython {
    param([string[]]$CommandParts)

    try {
        $command = $CommandParts[0]
        $arguments = @()
        if ($CommandParts.Length -gt 1) {
            $arguments = $CommandParts[1..($CommandParts.Length - 1)]
        }

        & $command @arguments -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Resolve-PythonExecutablePath {
    param([string[]]$CommandParts)
    return Invoke-PythonCapture -CommandParts $CommandParts -Code "import sys; print(sys.executable)"
}

function Test-PythonModuleAvailable {
    param(
        [string]$PythonExecutable,
        [string]$ModuleName
    )

    try {
        & $PythonExecutable -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$ModuleName') else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Test-PythonRuntimeHealthy {
    param([string]$PythonExecutable)

    try {
        & $PythonExecutable -c "import pyexpat, ssl" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Get-PythonBasePrefix {
    param([string]$PythonExecutable)

    $prefix = Invoke-PythonCapture -CommandParts @($PythonExecutable) -Code "import sys; print(sys.base_prefix)"
    if (-not $prefix) {
        throw "Unable to determine sys.base_prefix for: $PythonExecutable"
    }

    return $prefix
}

function Get-DefaultPythonInstallPaths {
    $paths = @()

    if ($env:LOCALAPPDATA) {
        $paths += Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
    }

    $programFiles = [Environment]::GetFolderPath("ProgramFiles")
    if ($programFiles) {
        $paths += Join-Path $programFiles "Python313\python.exe"
    }

    $programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
    if ($programFilesX86) {
        $paths += Join-Path $programFilesX86 "Python313\python.exe"
    }

    return $paths | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
}

function Get-SeedPythonExecutable {
    param(
        [switch]$PreferExternal
    )

    if ((-not $PreferExternal) -and (Test-CompatiblePython -CommandParts @($RepoInterpreterPython))) {
        return $RepoInterpreterPython
    }

    if ($PythonExe) {
        $candidate = @($PythonExe)
        if (Test-CompatiblePython -CommandParts $candidate) {
            $resolved = Resolve-PythonExecutablePath -CommandParts $candidate
            if ($resolved) {
                return $resolved
            }
        }
        throw "The supplied Python executable is not Python $RequiredPythonVersion."
    }

    $candidates = @(
        @("python"),
        @("py", "-$RequiredPythonVersion")
    )
    $checkedHints = @("python", "py -$RequiredPythonVersion")

    foreach ($path in Get-DefaultPythonInstallPaths) {
        $candidates += ,@($path)
        $checkedHints += $path
    }

    foreach ($candidate in $candidates) {
        if (Test-CompatiblePython -CommandParts $candidate) {
            $resolved = Resolve-PythonExecutablePath -CommandParts $candidate
            if ($resolved) {
                return $resolved
            }
        }
    }

    throw (
        "Python $RequiredPythonVersion was not found. " +
        "The bundled OpenVSP binary in this repository requires Python $RequiredPythonVersion. " +
        "Install Python $RequiredPythonVersion and run this script again.`n`n" +
        "Checked candidates:`n - $($checkedHints -join "`n - ")`n`n" +
        "If Python $RequiredPythonVersion is already installed elsewhere, rerun with:`n" +
        "run_project.bat setup -PythonExe ""C:\path\to\python.exe"""
    )
}

function Get-RecordedSourcePythonExecutable {
    $metadataPath = Join-Path $RepoInterpreterDir "repo_runtime.json"
    if (-not (Test-Path $metadataPath)) {
        return ""
    }

    try {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        $candidate = [string]$metadata.source_python
        if ($candidate -and (Test-CompatiblePython -CommandParts @($candidate))) {
            return $candidate
        }
    }
    catch {
        return ""
    }

    return ""
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-RepoCommandEnvironment {
    Ensure-Directory -Path $RepoTempDir
    Ensure-Directory -Path $RepoPipCacheDir

    return @{
        "TEMP" = $RepoTempDir
        "TMP" = $RepoTempDir
        "PIP_CACHE_DIR" = $RepoPipCacheDir
    }
}

function Write-RepoBootstrapPaths {
    param([string]$SitePackagesDir)

    Ensure-Directory -Path $SitePackagesDir
    Ensure-Directory -Path $RepoLibsDir

    $pthFile = Join-Path $SitePackagesDir $RepoPathBootstrapFileName
    $bootstrapLines = @(
        $RepoRoot,
        $RepoLibsDir
    )
    Set-Content -Path $pthFile -Value $bootstrapLines -Encoding ASCII
}

function Ensure-PipAvailable {
    param(
        [string]$PythonExecutable,
        [string]$RuntimeDescription
    )

    if (Test-PythonModuleAvailable -PythonExecutable $PythonExecutable -ModuleName "pip") {
        return
    }

    Invoke-External `
        -CommandParts @($PythonExecutable, $BootstrapPipScript) `
        -Description "Bootstrapping pip in $RuntimeDescription" `
        -EnvironmentOverrides (Get-RepoCommandEnvironment)

    if (-not (Test-PythonModuleAvailable -PythonExecutable $PythonExecutable -ModuleName "pip")) {
        throw "pip is still unavailable in $RuntimeDescription after running ensurepip."
    }
}

function Ensure-RepoInterpreterExists {
    $existingInterpreterReady = (Test-CompatiblePython -CommandParts @($RepoInterpreterPython))
    if ($existingInterpreterReady) {
        Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $RepoInterpreterDir "Lib\site-packages")
        if (Test-PythonRuntimeHealthy -PythonExecutable $RepoInterpreterPython) {
            Ensure-PipAvailable -PythonExecutable $RepoInterpreterPython -RuntimeDescription "the repository-local interpreter"
            return $RepoInterpreterPython
        }

        Write-Step "Repairing repository-local Python runtime dependencies"
    }

    Ensure-Directory -Path $RepoInterpreterDir
    $seedPython = ""
    if ($existingInterpreterReady) {
        $seedPython = Get-RecordedSourcePythonExecutable
        if (-not $seedPython) {
            $seedPython = Get-SeedPythonExecutable -PreferExternal
        }
    }
    else {
        $seedPython = Get-SeedPythonExecutable
    }

    if ($seedPython -eq $RepoInterpreterPython) {
        throw (
            "Unable to repair the repository-local interpreter because no external Python $RequiredPythonVersion " +
            "installation could be found. Rerun setup with -PythonExe pointing to a working Python $RequiredPythonVersion install."
        )
    }

    $sourceHome = Get-PythonBasePrefix -PythonExecutable $seedPython

    Invoke-External `
        -CommandParts @($seedPython, $BootstrapRuntimeScript, "--source-home", $sourceHome, "--target-home", $RepoInterpreterDir, "--source-python", $seedPython) `
        -Description "Creating repository-local Python runtime in interpreter"

    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $RepoInterpreterDir "Lib\site-packages")

    if (-not (Test-CompatiblePython -CommandParts @($RepoInterpreterPython))) {
        throw (
            "The repository-local interpreter could not be prepared successfully. " +
            "Check interpreter\python.exe and rerun setup."
        )
    }

    if (-not (Test-PythonRuntimeHealthy -PythonExecutable $RepoInterpreterPython)) {
        throw (
            "The repository-local interpreter was created, but core runtime DLLs are still missing. " +
            "Check interpreter\python.exe and rerun setup."
        )
    }

    Ensure-PipAvailable -PythonExecutable $RepoInterpreterPython -RuntimeDescription "the repository-local interpreter"

    return $RepoInterpreterPython
}

function Ensure-VenvExists {
    $repoPython = Ensure-RepoInterpreterExists

    if (Test-Path $VenvPython) {
        if (-not (Test-CompatiblePython -CommandParts @($VenvPython))) {
            throw (
                "The existing .venv does not use Python $RequiredPythonVersion. " +
                "Delete .venv and run run_project.bat setup again."
            )
        }
        Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $VenvDir "Lib\site-packages")
        Ensure-PipAvailable -PythonExecutable $VenvPython -RuntimeDescription "the local virtual environment"
        return
    }

    $command = @($repoPython, "-m", "venv", "--copies", $VenvDir)
    Invoke-External -CommandParts $command -Description "Creating local virtual environment in .venv"
    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $VenvDir "Lib\site-packages")
    Ensure-PipAvailable -PythonExecutable $VenvPython -RuntimeDescription "the local virtual environment"
}

function Install-ProjectDependencies {
    Ensure-RepoInterpreterExists | Out-Null
    Ensure-VenvExists

    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $RepoInterpreterDir "Lib\site-packages")
    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $VenvDir "Lib\site-packages")

    $repoCommandEnvironment = Get-RepoCommandEnvironment

    Invoke-External `
        -CommandParts @($VenvPython, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") `
        -Description "Updating pip tooling" `
        -EnvironmentOverrides $repoCommandEnvironment
    Invoke-External `
        -CommandParts @($VenvPython, "-m", "pip", "install", "--upgrade", "--target", $RepoLibsDir, "-r", $RequirementsFile) `
        -Description "Installing project dependencies into libs" `
        -EnvironmentOverrides $repoCommandEnvironment

    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $RepoInterpreterDir "Lib\site-packages")
    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $VenvDir "Lib\site-packages")

    Invoke-External -CommandParts @($VenvPython, "-m", "ipykernel", "install", "--user", "--name", $KernelName, "--display-name", $KernelDisplayName) -Description "Registering the project Jupyter kernel"
}

function Ensure-Setup {
    $libsReady = $false
    if (Test-Path $RepoLibsDir) {
        $libsReady = [bool](Get-ChildItem -LiteralPath $RepoLibsDir -Force | Where-Object { $_.Name -ne ".gitkeep" } | Select-Object -First 1)
    }

    if (
        (-not (Test-CompatiblePython -CommandParts @($RepoInterpreterPython))) -or
        (-not (Test-CompatiblePython -CommandParts @($VenvPython))) -or
        (-not $libsReady)
    ) {
        Install-ProjectDependencies
        return
    }

    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $RepoInterpreterDir "Lib\site-packages")
    Write-RepoBootstrapPaths -SitePackagesDir (Join-Path $VenvDir "Lib\site-packages")
}

function Run-Verification {
    Ensure-Setup
    Invoke-External -CommandParts @($VenvPython, $VerifyScript) -Description "Running automatic environment verification"
}

switch ($Action) {
    "setup" {
        Install-ProjectDependencies
        Run-Verification
        Write-Step "Setup completed. You can now use run_project.bat to launch the notebook or app."
    }
    "verify" {
        Run-Verification
    }
    "streamlit" {
        Run-Verification
        Invoke-External -CommandParts @($VenvPython, "-m", "streamlit", "run", $StreamlitApp) -Description "Launching the Streamlit interface"
    }
    "notebook" {
        Run-Verification
        Write-Step "Launching JupyterLab. If prompted for a kernel, select '$KernelDisplayName'."
        Invoke-External -CommandParts @($VenvPython, "-m", "jupyter", "lab", $NotebookPath) -Description "Starting JupyterLab"
    }
}
