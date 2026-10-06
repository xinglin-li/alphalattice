# Locked product browser-operation entry, also used for development and E2E.
# Usage: ./scripts/playwright.ps1 setup|check|run [playwright-cli arguments...]
$ErrorActionPreference = 'Stop'
$toolRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\third_party\playwright'))
$browserRoot = Join-Path $toolRoot '.browsers'
$packagePath = Join-Path $toolRoot 'node_modules\@playwright\cli\package.json'
$cliPath = Join-Path $toolRoot 'node_modules\@playwright\cli\playwright-cli.js'
$playwrightPath = Join-Path $toolRoot 'node_modules\playwright\cli.js'
$browserManifest = Join-Path $toolRoot 'node_modules\playwright-core\browsers.json'
$dependencyManifest = Get-Content -Raw -LiteralPath (Join-Path $toolRoot 'package.json') | ConvertFrom-Json
$expectedCliVersion = $dependencyManifest.dependencies.'@playwright/cli'
$expectedPlaywrightVersion = $dependencyManifest.dependencies.playwright
$action = if ($args.Count -gt 0) { [string]$args[0] } else { 'check' }
$forward = [string[]]@($args | Select-Object -Skip 1)

function Fail([string]$message) {
    [Console]::Error.WriteLine("playwright-tool: $message")
    exit 2
}

function Read-Installation {
    if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $cliPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $playwrightPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $browserManifest -PathType Leaf)) {
        Fail "Pinned CLI is missing from $toolRoot. Run ./scripts/playwright.ps1 setup."
    }
    $package = Get-Content -Raw -LiteralPath $packagePath | ConvertFrom-Json
    if ($package.version -ne $expectedCliVersion) {
        Fail "Expected @playwright/cli $expectedCliVersion; found $($package.version). Run ./scripts/playwright.ps1 setup."
    }
    $library = Get-Content -Raw -LiteralPath (Join-Path $toolRoot 'node_modules\playwright\package.json') | ConvertFrom-Json
    $core = Get-Content -Raw -LiteralPath (Join-Path $toolRoot 'node_modules\playwright-core\package.json') | ConvertFrom-Json
    if ($library.version -ne $expectedPlaywrightVersion -or $core.version -ne $expectedPlaywrightVersion) {
        Fail "Expected Playwright and playwright-core $expectedPlaywrightVersion. Run ./scripts/playwright.ps1 setup."
    }
    $manifest = Get-Content -Raw -LiteralPath $browserManifest | ConvertFrom-Json
    $browser = @($manifest.browsers | Where-Object { $_.name -eq 'chromium-headless-shell' })
    if ($browser.Count -ne 1) { Fail "Pinned browser manifest is invalid: $browserManifest" }
    $executable = Join-Path $browserRoot "chromium_headless_shell-$($browser[0].revision)\chrome-headless-shell-win64\chrome-headless-shell.exe"
    return [pscustomobject]@{ Browser = $browser[0]; Executable = $executable }
}

try {
    $node = Get-Command node.exe -ErrorAction Stop
    $npm = Get-Command npm.cmd -ErrorAction Stop
} catch {
    Fail 'Node.js >=20 and npm are required. Install them, then run ./scripts/playwright.ps1 setup.'
}
$nodeVersion = (& $node.Source --version).Trim()
if ($LASTEXITCODE -ne 0 -or [int]($nodeVersion.TrimStart('v').Split('.')[0]) -lt 20) {
    Fail "Node.js >=20 is required; found $nodeVersion."
}
$npmVersion = (& $npm.Source --version).Trim()
if ($LASTEXITCODE -ne 0) { Fail 'npm failed. Repair Node.js/npm, then rerun setup.' }

# This pinned CLI explicitly supports NO_UPDATE_NOTIFIER. Browser files stay local.
$env:NO_UPDATE_NOTIFIER = '1'
$env:PLAYWRIGHT_BROWSERS_PATH = $browserRoot

switch ($action) {
    'setup' {
        if ($forward.Count -ne 0) { Fail 'setup takes no additional arguments.' }
        if (-not (Test-Path -LiteralPath (Join-Path $toolRoot 'package-lock.json') -PathType Leaf)) {
            Fail "Lockfile is missing from $toolRoot."
        }
        & $npm.Source ci --prefix $toolRoot --omit=dev --ignore-scripts --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $installed = Read-Installation
        if (-not (Test-Path -LiteralPath $installed.Executable -PathType Leaf)) {
            & $node.Source $playwrightPath install chromium --only-shell
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        & $node.Source $cliPath --version
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not (Test-Path -LiteralPath $installed.Executable -PathType Leaf)) {
            Fail "Chromium headless shell installation did not create $($installed.Executable)."
        }
        Write-Output "Pinned Chromium $($installed.Browser.browserVersion), revision $($installed.Browser.revision): $($installed.Executable)"
    }
    'check' {
        if ($forward.Count -ne 0) { Fail 'check takes no additional arguments.' }
        $installed = Read-Installation
        Write-Output "Node: $nodeVersion ($($node.Source))"
        Write-Output "npm: $npmVersion ($($npm.Source))"
        Write-Output "CLI: $expectedCliVersion ($cliPath)"
        Write-Output "Playwright: $expectedPlaywrightVersion"
        & $node.Source $cliPath --version
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not (Test-Path -LiteralPath $installed.Executable -PathType Leaf)) {
            Fail "Pinned Chromium $($installed.Browser.browserVersion) revision $($installed.Browser.revision) is missing at $($installed.Executable). Run ./scripts/playwright.ps1 setup."
        }
        Write-Output "Browser: Chromium $($installed.Browser.browserVersion), revision $($installed.Browser.revision) ($($installed.Executable))"
    }
    'run' {
        if ($forward.Count -eq 0) { Fail 'run requires playwright-cli arguments; e.g. run -s=my-task snapshot.' }
        $installed = Read-Installation
        if (-not (Test-Path -LiteralPath $installed.Executable -PathType Leaf)) {
            Fail "Pinned Chromium is missing at $($installed.Executable). Run ./scripts/playwright.ps1 setup."
        }
        & $node.Source $cliPath @forward
        exit $LASTEXITCODE
    }
    default { Fail 'Use setup, check, or run [playwright-cli arguments...].' }
}
