<#
.SYNOPSIS
    Deploys the Phi-3 ONNX model to an Xbox console via Xbox Device Portal.

.DESCRIPTION
    Uploads all model files from the local Assets/Model directory to the Xbox's
    LocalState folder (DavidModel/directml-int4-awq-block-128/).

    Credentials are prompted interactively via Get-Credential — nothing is
    stored on disk or committed to source control.

    Run this from the Windows VM after enabling Xbox Device Portal on the console.

.PARAMETER XboxIP
    IP address of the Xbox (e.g. "192.168.1.50"). If omitted, you will be prompted.

.EXAMPLE
    .\DeployModel.ps1
    .\DeployModel.ps1 -XboxIP 192.168.1.50
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$XboxIP
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

$PackageFamilyName  = "David.XboxAIServer_8wekyb3d8bbwe"   # Update if your PFN differs
$LocalStateSubPath  = "DavidModel/directml-int4-awq-block-128"
$ScriptDir          = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ModelSourceDir     = Join-Path $ScriptDir "Assets\Model\directml\directml-int4-awq-block-128"

# ──────────────────────────────────────────────
#  Validate source model exists
# ──────────────────────────────────────────────

if (-not (Test-Path $ModelSourceDir)) {
    Write-Error "Model source directory not found: $ModelSourceDir`nRun download_phi3.py first."
    exit 1
}

$modelFiles = Get-ChildItem -Path $ModelSourceDir -File
if ($modelFiles.Count -eq 0) {
    Write-Error "No model files found in: $ModelSourceDir"
    exit 1
}

# ──────────────────────────────────────────────
#  Prompt for Xbox IP and credentials
# ──────────────────────────────────────────────

if ([string]::IsNullOrWhiteSpace($XboxIP)) {
    $XboxIP = Read-Host "Enter Xbox IP address"
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Project David — Model Deployer" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Xbox:    $XboxIP"
Write-Host "  Source:  $ModelSourceDir"
Write-Host "  Target:  LocalState\$LocalStateSubPath"
Write-Host ""

# Interactive credential prompt — never saved to disk
$cred = Get-Credential -Message "Xbox Device Portal credentials (see Dev Home > Remote Access)"

# ──────────────────────────────────────────────
#  Bypass self-signed certificate (Xbox Dev Portal uses HTTPS with self-signed cert)
# ──────────────────────────────────────────────

if (-not ([System.Management.Automation.PSTypeName]'TrustAllCertsPolicy').Type) {
    Add-Type @"
using System.Net;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(
        ServicePoint srvPoint, X509Certificate certificate,
        WebRequest request, int certificateProblem) { return true; }
}
"@
}
[System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

# ──────────────────────────────────────────────
#  Upload each file via Device Portal
# ──────────────────────────────────────────────

$baseUri = "https://${XboxIP}:11443/api/filesystem/apps/file"

$totalFiles  = $modelFiles.Count
$currentFile = 0

foreach ($file in $modelFiles) {
    $currentFile++
    $sizeMB     = [math]::Round($file.Length / 1MB, 1)
    $remotePath = $LocalStateSubPath + "/" + $file.Name

    Write-Host "[$currentFile/$totalFiles] Uploading $($file.Name) ($sizeMB MB)..." -ForegroundColor Yellow

    # Build the query parameters for the Device Portal file API
    $queryParams = @{
        knownfolderid = "LocalAppData"
        packagefullname = $PackageFamilyName
        path          = "\LocalState\$remotePath" -replace "/", "\"
    }

    $uri = $baseUri + "?" + (($queryParams.GetEnumerator() | ForEach-Object {
        "$($_.Key)=$([System.Uri]::EscapeDataString($_.Value))"
    }) -join "&")

    try {
        $fileBytes = [System.IO.File]::ReadAllBytes($file.FullName)

        Invoke-RestMethod `
            -Uri $uri `
            -Method POST `
            -Credential $cred `
            -Body $fileBytes `
            -ContentType "application/octet-stream" `
            -TimeoutSec 600

        Write-Host "  ✅ $($file.Name) uploaded successfully." -ForegroundColor Green
    }
    catch {
        Write-Host "  ❌ Failed to upload $($file.Name): $_" -ForegroundColor Red

        if ($file.Name -eq "model.onnx.data") {
            Write-Host ""
            Write-Host "  TIP: The 2GB model.onnx.data may be too large for Invoke-RestMethod." -ForegroundColor Magenta
            Write-Host "  Alternative: Use Xbox Device Portal web UI to upload manually:" -ForegroundColor Magenta
            Write-Host "    1. Open https://${XboxIP}:11443 in your browser" -ForegroundColor Magenta
            Write-Host "    2. Navigate to File Explorer > LocalAppData" -ForegroundColor Magenta
            Write-Host "    3. Find: $PackageFamilyName\LocalState" -ForegroundColor Magenta
            Write-Host "    4. Create folder: $LocalStateSubPath" -ForegroundColor Magenta
            Write-Host "    5. Upload model.onnx.data into that folder" -ForegroundColor Magenta
            Write-Host ""
        }
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "  Launch the David AI Server app on the Xbox." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
