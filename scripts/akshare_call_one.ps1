[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$BaseUrl = "http://10.147.20.211:1818/mcp",
  [string]$ApiName = "stock_bid_ask_em",
  [string]$Symbol = "002385",
  [string]$Period = "",
  [string]$Adjust = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-McpSessionId {
  param([string]$Url)
  $headersText = & curl.exe -s -D - -o NUL --noproxy "*" -H "Accept: text/event-stream" $Url
  $line = ($headersText | Select-String -Pattern '^(?i)mcp-session-id:\s*' | Select-Object -First 1)
  if (-not $line) { throw "Mcp-Session-Id header not found from $Url" }
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
    $json = $BodyObject | ConvertTo-Json -Depth 64 -Compress
    Set-Content -NoNewline -Encoding utf8 $tmp $json
    $raw = & curl.exe -s --noproxy "*" -m 180 `
      -H "Accept: application/json, text/event-stream" `
      -H "Content-Type: application/json" `
      -H "Mcp-Session-Id: $SessionId" `
      --data-binary "@$tmp" `
      $Url
    $dataLine = ($raw -split "`n" | Where-Object { $_ -match '^\s*data:\s*' } | Select-Object -Last 1)
    if (-not $dataLine) { throw "No 'data:' line found in response. Raw=`n$raw" }
    $payload = ($dataLine -replace '^\s*data:\s*', '').Trim()
    return ($payload | ConvertFrom-Json)
  }
  finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp
  }
}

function Invoke-AkshareQuery {
  param(
    [string]$Url,
    [string]$SessionId,
    [string]$ApiName,
    [hashtable]$Params
  )
  $req = @{
    jsonrpc = "2.0"
    id      = 1
    method  = "tools/call"
    params  = @{
      name      = "execute_akshare_query"
      arguments = @{
        api_name = $ApiName
        params   = $Params
      }
    }
  }
  return (Invoke-McpSseRequest -Url $Url -SessionId $SessionId -BodyObject $req)
}

$sid = Get-McpSessionId -Url $BaseUrl
$null = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 1
  method  = "initialize"
  params  = @{
    protocolVersion = "2024-11-05"
    clientInfo      = @{ name = "codex-cli"; version = "call-one" }
    capabilities    = @{ tools = @{}; resources = @{}; prompts = @{} }
  }
}

$params = @{ symbol = $Symbol }
if ($Period) { $params["period"] = $Period }
if ($PSBoundParameters.ContainsKey("Adjust")) { $params["adjust"] = $Adjust }

$resp = Invoke-AkshareQuery -Url $BaseUrl -SessionId $sid -ApiName $ApiName -Params $params
$resp | ConvertTo-Json -Depth 64
