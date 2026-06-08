[CmdletBinding()]
param(
    [string]$Configuration = 'Release',
    [string]$OutputName    = 'Diffsinger-SSM',
    [switch]$Test
)

# Build the SSM extension and pack it as a .tlx (zip) ready to drop into TuneLab.
#
# The .tlx layout follows what TuneLab v1.6 expects:
#   <root>/description.json       — extension manifest (already in build output)
#   <root>/Diffsinger-SSM.dll
#
# We deliberately do NOT bundle TuneLab.Base / TuneLab.Extensions.Voices /
# Diffsinger3rdApi / ChoristaVocoder / YamlDotNet / Microsoft.ML.OnnxRuntime115 —
# those are provided at runtime by TuneLab itself or by the already-installed
# ChoristaDiffsinger.tlx.  Including them would cause "type defined in two assemblies"
# errors when TuneLab loads our tlx after Choristad's.

$ErrorActionPreference = 'Stop'
$root      = Split-Path -Parent $PSCommandPath
$repo      = Resolve-Path (Join-Path $root '..')
$proj      = Join-Path $root 'src/ChoristaDiffsinger-SSM/ChoristaDiffsinger-SSM.csproj'
$testsProj = Join-Path $root 'tests/ChoristaDiffsinger-SSM.Tests/ChoristaDiffsinger-SSM.Tests.csproj'
$outDir    = Join-Path $root 'src/ChoristaDiffsinger-SSM/bin' | Join-Path -ChildPath $Configuration | Join-Path -ChildPath 'net8.0'
$dist      = Join-Path $root 'dist'
$staging   = Join-Path $dist 'staging'
$tlx       = Join-Path $dist ($OutputName + '.tlx')

Push-Location $root
try {
    Write-Host "[pack] dotnet build ($Configuration)..."
    dotnet build $proj -c $Configuration -nologo -v:minimal | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "dotnet build failed (exit=$LASTEXITCODE)" }

    if ($Test) {
        Write-Host "[pack] dotnet test ($Configuration)..."
        dotnet test $testsProj -c $Configuration --nologo | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "dotnet test failed (exit=$LASTEXITCODE)" }
    }

    if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
    New-Item -ItemType Directory -Path $staging | Out-Null

    Copy-Item (Join-Path $outDir 'description.json')               (Join-Path $staging 'description.json')
    Copy-Item (Join-Path $outDir 'Diffsinger-SSM.dll')              (Join-Path $staging 'Diffsinger-SSM.dll')

    if (Test-Path $tlx) { Remove-Item -Force $tlx }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($staging, $tlx)

    Write-Host ("[pack] {0}" -f $tlx)
    Get-ChildItem $tlx | Format-List | Out-Host
}
finally {
    Pop-Location
}