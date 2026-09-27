# ============================================================
# CYCLO-VISION — admin one-shot fix (run in an ELEVATED PowerShell)
# ------------------------------------------------------------
# Fixes the two issues found in the CI logs:
#   1. Stale machine-wide CURL_CA_BUNDLE that points at a file that
#      no longer exists -> every plain `requests` HTTPS call machine-wide
#      fails with "Could not find a suitable TLS CA certificate bundle".
#      (The app itself is shielded by backend/app/core/net.py; external
#      scripts such as scripts/verify_full_stack.py are not.)
#   2. PostgreSQL service 'postgresql-x64-18' is stopped -> the API logs
#      "Database unavailable at startup" and skips history persistence.
#
# Usage (from your normal terminal):
#   Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File "<repo>\logs\admin_fix.ps1"'
# OR open PowerShell "as Administrator" yourself and run:
#   powershell -ExecutionPolicy Bypass -File "<repo>\logs\admin_fix.ps1"
#
# Safe to re-run. The old CA value is saved to
# logs\env_backup_machine_ca.txt before anything is changed.
# ============================================================
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Continue'

$backupFile = Join-Path $PSScriptRoot 'env_backup_machine_ca.txt'
$caName = 'CURL_CA_BUNDLE'
$svcName = 'postgresql-x64-18'

Write-Host '=== CYCLO-VISION admin fix ===' -ForegroundColor Cyan

# --- 1. stale machine-wide CA bundle -------------------------------------
$current = [Environment]::GetEnvironmentVariable($caName, 'Machine')
if ($current) {
    Add-Content -Path $backupFile -Value ('{0}  {1}={2}' -f (Get-Date -Format s), $caName, $current)
    Write-Host "Backed up machine-wide $caName to $backupFile (value: $current)"
    if (Test-Path -LiteralPath $current) {
        Write-Host "The configured CA file exists on disk; leaving the variable in place."
    } else {
        [Environment]::SetEnvironmentVariable($caName, $null, 'Machine')
        Write-Host "REMOVED stale machine-wide $caName (was: $current)" -ForegroundColor Green
        Write-Host 'Open a NEW terminal for the change to take effect in it.'
    }
} else {
    Write-Host "Machine-wide $caName already clear - nothing to do." -ForegroundColor Green
}

# --- 2. PostgreSQL service ------------------------------------------------
$svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    Write-Host "Service '$svcName' not found on this machine." -ForegroundColor Yellow
} elseif ($svc.Status -eq 'Running') {
    Write-Host "PostgreSQL service '$svcName' is already running." -ForegroundColor Green
} else {
    try {
        Start-Service -Name $svcName -ErrorAction Stop
        Start-Sleep -Seconds 3
        $svc.Refresh()
        Write-Host "PostgreSQL service status: $($svc.Status)" -ForegroundColor Green
    } catch {
        Write-Host "Failed to start '$svcName': $($_.Exception.Message)" -ForegroundColor Red
        Write-Host 'Check the service log:  sc.exe query ' + $svcName
    }
}
Write-Host ''
sc.exe query $svcName

# --- 3. next steps ---------------------------------------------------------
Write-Host ''
Write-Host '=== Next steps (normal terminal, repo root) ===' -ForegroundColor Cyan
Write-Host ' a) Native PostgreSQL 18: ensure the app role/database exist (run once,'
Write-Host '    it will ask for your postgres superuser password):'
Write-Host '      & "D:\PostgreSQL17\bin\psql.exe" -U postgres -c "CREATE ROLE cyclo LOGIN PASSWORD ''cyclo_pass'';"'
Write-Host '      & "D:\PostgreSQL17\bin\psql.exe" -U postgres -c "CREATE DATABASE cyclo_vision OWNER cyclo;"'
Write-Host '    (If that path does not exist, find psql.exe under your PostgreSQL install.)'
Write-Host ' b) Or use Docker instead (matches the default DATABASE_URL):'
Write-Host '      docker compose up -d db'
Write-Host ' c) Create the tables:'
Write-Host '      python scripts\init_db.py'
Write-Host ' d) Restart the backend, then check:  curl.exe http://127.0.0.1:8000/api/health'
