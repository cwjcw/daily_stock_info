[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$BaseUrl = "http://10.147.20.211:1818/mcp",
  [string]$ListPath = "",
  [string[]]$Tickers = @(),
  [string]$OutputDir = "output",
  [string]$OutputPrefix = "realtime_quotes"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function U {
  param([int[]]$Codes)
  return -join ($Codes | ForEach-Object { [char]$_ })
}

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
    id      = (Get-Random -Minimum 1000 -Maximum 999999)
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

function Parse-TickerLine {
  param([string]$Line)

  $t = $Line.Trim()
  if (-not $t) { return $null }
  if ($t.StartsWith("#")) { return $null }

  $parts = $t -split '\s+', 2
  $code = $parts[0].Trim()
  $rest = if ($parts.Length -gt 1) { $parts[1].Trim() } else { "" }

  $name = ""
  if ($rest) {
    $name = ($rest -split "[\\uFF08(]", 2)[0].Trim()
  }

  return [pscustomobject]@{
    ts_code = $code
    name    = $name
    raw     = $t
  }
}

function TsCodeToSymbol {
  param([string]$TsCode)
  return $TsCode.Split(".")[0]
}

function TsCodeToMarketPrefix {
  param([string]$TsCode)
  if ($TsCode -match '\\.SZ$') { return "sz" }
  if ($TsCode -match '\\.SH$') { return "sh" }
  if ($TsCode -match '\\.BJ$') { return "bj" }
  return ""
}

function TsCodeToXqSymbol {
  param([string]$TsCode)
  $parts = $TsCode.Split(".")
  $code = $parts[0]
  $suffix = if ($parts.Length -gt 1) { $parts[1].ToUpperInvariant() } else { "" }
  if ($suffix) { return "${suffix}${code}" }
  return $code
}

Ensure-Dir $OutputDir

$items = New-Object System.Collections.Generic.List[object]
if ($ListPath) {
  if (-not (Test-Path -LiteralPath $ListPath)) { throw "ListPath not found: $ListPath" }
  Get-Content -LiteralPath $ListPath -Encoding utf8 | ForEach-Object {
    $p = Parse-TickerLine -Line $_
    if ($p) { $items.Add($p) }
  }
}
foreach ($t in $Tickers) {
  $p = Parse-TickerLine -Line $t
  if ($p) { $items.Add($p) }
}
if ($items.Count -eq 0) { throw "No tickers provided. Use -ListPath or -Tickers." }

$sid = Get-McpSessionId -Url $BaseUrl
$null = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 1
  method  = "initialize"
  params  = @{
    protocolVersion = "2024-11-05"
    clientInfo      = @{ name = "codex-cli"; version = "realtime-batch" }
    capabilities    = @{ tools = @{}; resources = @{}; prompts = @{} }
  }
}

# Labels used by stock_individual_spot_xq (built dynamically to keep this script ASCII-only)
$kCode = U @(0x4EE3, 0x7801)
$kName = U @(0x540D, 0x79F0)
$kExchange = U @(0x4EA4, 0x6613, 0x6240)
$kTime = U @(0x65F6, 0x95F4)
$kPrice = U @(0x73B0, 0x4EF7)
$kPct = U @(0x6DA8, 0x5E45)
$kChg = U @(0x6DA8, 0x8DCC)
$kOpen = U @(0x4ECA, 0x5F00)
$kPrev = U @(0x6628, 0x6536)
$kHigh = U @(0x6700, 0x9AD8)
$kLow = U @(0x6700, 0x4F4E)
$kVol = U @(0x6210, 0x4EA4, 0x91CF)
$kAmt = U @(0x6210, 0x4EA4, 0x989D)
$kTurn = U @(0x5468, 0x8F6C, 0x7387)
$kAvg = U @(0x5747, 0x4EF7)
$kCurrency = U @(0x8D27, 0x5E01)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outCsv = Join-Path $OutputDir "${OutputPrefix}_${stamp}.csv"
$outJson = Join-Path $OutputDir "${OutputPrefix}_${stamp}.json"
$latestCsv = Join-Path $OutputDir "${OutputPrefix}_latest.csv"
$latestJson = Join-Path $OutputDir "${OutputPrefix}_latest.json"

$rows = New-Object System.Collections.Generic.List[object]
$fail = New-Object System.Collections.Generic.List[object]

foreach ($it in $items) {
  $symbol = TsCodeToSymbol -TsCode $it.ts_code
  $xqSymbol = TsCodeToXqSymbol -TsCode $it.ts_code
  try {
    $apiUsed = "stock_individual_spot_xq"
    $kv = @{}

    $lastPrice = $null
    $pctChange = $null
    $change = $null
    $open = $null
    $prevClose = $null
    $high = $null
    $low = $null
    $volumeHand = $null
    $amountCny = $null
    $turnoverPct = $null
    $volRatio = $null
    $bid1 = $null
    $bid1Vol = $null
    $ask1 = $null
    $ask1Vol = $null

    $resp = Invoke-AkshareQuery -Url $BaseUrl -SessionId $sid -ApiName $apiUsed -Params @{ symbol = $xqSymbol }
    $sc = $resp.result.structuredContent
    if (-not $sc.ok) { throw "akshare returned ok=false" }

    foreach ($pair in @($sc.data)) {
      if ($null -ne $pair.item -and -not $kv.ContainsKey($pair.item)) {
        $kv[$pair.item] = $pair.value
      }
    }

    $nameFromApi = $kv[$kName]
    if ($nameFromApi) { $it.name = $nameFromApi }

    $lastPrice = $kv[$kPrice]
    $pctChange = $kv[$kPct]
    $change = $kv[$kChg]
    $open = $kv[$kOpen]
    $prevClose = $kv[$kPrev]
    $high = $kv[$kHigh]
    $low = $kv[$kLow]
    $volumeHand = $kv[$kVol]
    $amountCny = $kv[$kAmt]
    $turnoverPct = $kv[$kTurn]
    $avgPrice = $kv[$kAvg]
    $currency = $kv[$kCurrency]
    $ts = $kv[$kTime]

    $rows.Add([pscustomobject]([ordered]@{
      ts_code      = $it.ts_code
      name         = $it.name
      symbol       = $symbol
      xq_symbol    = $xqSymbol
      api_used     = $apiUsed
      fetched_at   = if ($ts) { $ts } else { (Get-Date).ToString("yyyy-MM-dd HH:mm:ss") }
      last_price   = $lastPrice
      pct_change   = $pctChange
      change       = $change
      open         = $open
      prev_close   = $prevClose
      high         = $high
      low          = $low
      volume_hand  = $volumeHand
      amount_cny   = $amountCny
      turnover_pct = $turnoverPct
      avg_price    = $avgPrice
      currency     = $currency
      exchange     = $kv[$kExchange]
      bid1         = $bid1
      bid1_vol     = $bid1Vol
      ask1         = $ask1
      ask1_vol     = $ask1Vol
    }))
  }
  catch {
    $fail.Add([pscustomobject]@{
      ts_code = $it.ts_code
      name    = $it.name
      symbol  = $symbol
      error   = $_.Exception.Message
      raw     = $it.raw
    })
  }
}

$bundle = [ordered]@{
  ok = $true
  generated = (Get-Date).ToString("o")
  baseUrl = $BaseUrl
  sessionId = $sid
  total = $items.Count
  success = $rows.Count
  failed = $fail.Count
  rows = $rows
  failures = $fail
}

($bundle | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 $outJson
($bundle | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 $latestJson

$rows | Export-Csv -NoTypeInformation -Encoding utf8 $outCsv
$rows | Export-Csv -NoTypeInformation -Encoding utf8 $latestCsv

Write-Host "Akshare realtime batch done"
Write-Host "  Total:   $($items.Count)"
Write-Host "  Success: $($rows.Count)"
Write-Host "  Failed:  $($fail.Count)"
Write-Host "  CSV:     $outCsv"
Write-Host "  JSON:    $outJson"
if ($fail.Count -gt 0) {
  Write-Host "Failures:"
  $fail | Select-Object -First 20 | Format-Table -AutoSize | Out-String | Write-Host
}
