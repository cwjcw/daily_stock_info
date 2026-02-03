param(
  [string]$BaseUrl = "http://10.147.20.211:1818/mcp",
  [string]$Symbol = "sz002371",
  [string]$StartDate = "20260101",
  [string]$EndDate = "20260128",
  [string]$Adjust = "qfq",
  [string]$OutputDir = "output"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-McpSessionId {
  param([string]$Url)

  $headersText = & curl.exe -s -D - -o NUL --noproxy "*" -H "Accept: text/event-stream" $Url
  $line = ($headersText | Select-String -Pattern '^(?i)mcp-session-id:\s*' | Select-Object -First 1)
  if (-not $line) {
    throw "Mcp-Session-Id header not found from $Url"
  }
  return ($line.Line -split ":", 2)[1].Trim()
}

function Invoke-McpSseRequest {
  param(
    [string]$Url,
    [string]$SessionId,
    [object]$BodyObject
  )

  $tmp = New-TemporaryFile
  try {
    $json = $BodyObject | ConvertTo-Json -Depth 32 -Compress
    Set-Content -NoNewline -Encoding utf8 $tmp $json

    $raw = & curl.exe -s --noproxy "*" -m 120 `
      -H "Accept: application/json, text/event-stream" `
      -H "Content-Type: application/json" `
      -H "Mcp-Session-Id: $SessionId" `
      --data-binary "@$tmp" `
      $Url

    $dataLine = ($raw -split "`n" | Where-Object { $_ -match '^\s*data:\s*' } | Select-Object -Last 1)
    if (-not $dataLine) {
      throw "No 'data:' line found in response. Raw=`n$raw"
    }

    $payload = ($dataLine -replace '^\s*data:\s*', '').Trim()
    $obj = $payload | ConvertFrom-Json
    return $obj
  }
  finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp
  }
}

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

Ensure-Dir $OutputDir

$sid = Get-McpSessionId -Url $BaseUrl

$initialize = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 1
  method  = "initialize"
  params  = @{
    protocolVersion = "2024-11-05"
    clientInfo      = @{ name = "codex-cli"; version = "smoke-test" }
    capabilities    = @{ tools = @{}; resources = @{}; prompts = @{} }
  }
}

$toolsList = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 2
  method  = "tools/list"
  params  = @{}
}

$searchDocs = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 3
  method  = "tools/call"
  params  = @{
    name      = "search_api_docs"
    arguments = @{ keyword = "stock_zh_a_hist_tx"; limit = 5 }
  }
}

$sample = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 4
  method  = "tools/call"
  params  = @{
    name      = "execute_akshare_query"
    arguments = @{
      api_name = "stock_zh_a_hist_tx"
      params   = @{
        symbol     = $Symbol
        start_date = $StartDate
        end_date   = $EndDate
        adjust     = $Adjust
      }
    }
  }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outPath = Join-Path $OutputDir "akshare_mcp_smoke_$stamp.json"
$latestPath = Join-Path $OutputDir "akshare_mcp_smoke_latest.json"

$bundle = [ordered]@{
  ok         = $true
  generated  = (Get-Date).ToString("o")
  baseUrl    = $BaseUrl
  sessionId  = $sid
  initialize = $initialize
  tools_list = $toolsList
  search_api_docs = $searchDocs
  sample_call = $sample
}

($bundle | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 $outPath
($bundle | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 $latestPath

$toolCount = @($toolsList.result.tools).Count
$rowCount = $null
if ($sample.result.structuredContent.meta.returned_rows) {
  $rowCount = $sample.result.structuredContent.meta.returned_rows
}

Write-Host "Akshare MCP smoke test OK"
Write-Host "  SessionId: $sid"
Write-Host "  Tools:     $toolCount"
if ($rowCount -ne $null) {
  Write-Host "  Sample rows (stock_zh_a_hist_tx): $rowCount"
}
Write-Host "  Output:    $outPath"
