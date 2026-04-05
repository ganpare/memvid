param(
    [string]$ModelsDir = "",
    [string]$RepoId = "sungo-ganpare/memvid-embedding-models",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-DefaultModelsDir {
    if ($env:LOCALAPPDATA) {
        return Join-Path $env:LOCALAPPDATA "memvid\\text-models"
    }
    return Join-Path $HOME ".cache\\memvid\\text-models"
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile,
        [switch]$Force
    )

    if ((Test-Path $OutFile) -and -not $Force) {
        Write-Host "Skipping existing file: $OutFile"
        return
    }

    $parent = Split-Path -Parent $OutFile
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
}

if (-not $ModelsDir) {
    $ModelsDir = Get-DefaultModelsDir
}

$ModelsDir = [System.IO.Path]::GetFullPath($ModelsDir)
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$base = "https://huggingface.co/$RepoId/resolve/main"

$downloads = @(
    @{ Url = "$base/multilingual-e5-large/multilingual-e5-large.onnx"; Out = (Join-Path $ModelsDir "multilingual-e5-large.onnx") },
    @{ Url = "$base/multilingual-e5-large/model.onnx_data"; Out = (Join-Path $ModelsDir "model.onnx_data") },
    @{ Url = "$base/multilingual-e5-large/multilingual-e5-large_tokenizer.json"; Out = (Join-Path $ModelsDir "multilingual-e5-large_tokenizer.json") },
    @{ Url = "$base/ruri-pt-large/ruri-pt-large.onnx"; Out = (Join-Path $ModelsDir "ruri-pt-large.onnx") },
    @{ Url = "$base/ruri-pt-large/vocab.txt"; Out = (Join-Path $ModelsDir "vocab.txt") },
    @{ Url = "$base/ruri-pt-large/tokenizer_config.json"; Out = (Join-Path $ModelsDir "tokenizer_config.json") },
    @{ Url = "$base/ruri-pt-large/special_tokens_map.json"; Out = (Join-Path $ModelsDir "special_tokens_map.json") }
)

foreach ($item in $downloads) {
    Download-File -Url $item.Url -OutFile $item.Out -Force:$Force
}

Write-Host ""
Write-Host "Japanese embedding models are ready in: $ModelsDir"
Write-Host "Use TextEmbedConfig::multilingual_e5_large() or TextEmbedConfig::ruri_pt_large()."
