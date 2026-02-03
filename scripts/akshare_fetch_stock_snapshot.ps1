param(
  [string]$BaseUrl = "http://10.147.20.211:1818/mcp",
  [string]$TsCode = "002385.SZ",
  [string]$OutputDir = "output",
  [int]$NewsLimit = 30,
  [int]$AnnsLimit = 50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

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
    $json = $BodyObject | ConvertTo-Json -Depth 64 -Compress
    Set-Content -NoNewline -Encoding utf8 $tmp $json

    $raw = & curl.exe -s --noproxy "*" -m 180 `
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

function Search-AkshareDocs {
  param(
    [string]$Url,
    [string]$SessionId,
    [string]$Keyword,
    [int]$Limit = 10
  )

  $req = @{
    jsonrpc = "2.0"
    id      = (Get-Random -Minimum 1000 -Maximum 999999)
    method  = "tools/call"
    params  = @{
      name      = "search_api_docs"
      arguments = @{
        keyword = $Keyword
        limit   = $Limit
      }
    }
  }

  return (Invoke-McpSseRequest -Url $Url -SessionId $SessionId -BodyObject $req)
}

function Convert-TsCodeToAkSymbolCandidates {
  param([string]$Code)

  $parts = $Code.Split(".")
  $stock = $parts[0]
  $market = if ($parts.Length -gt 1) { $parts[1].ToUpperInvariant() } else { "" }

  $cands = New-Object System.Collections.Generic.List[string]
  $cands.Add($stock)
  if ($market -eq "SZ") { $cands.Add("sz$stock") }
  if ($market -eq "SH") { $cands.Add("sh$stock") }
  if ($market -eq "BJ") { $cands.Add("bj$stock") }
  $cands.Add($Code)
  return $cands.ToArray()
}

function Try-ExecuteWithSymbolCandidates {
  param(
    [string]$Url,
    [string]$SessionId,
    [string]$ApiName,
    [hashtable]$BaseParams,
    [string[]]$SymbolCandidates
  )

  $tries = @()
  foreach ($symbol in $SymbolCandidates) {
    foreach ($key in @("symbol", "code", "stock")) {
      $params = @{}
      foreach ($k in $BaseParams.Keys) { $params[$k] = $BaseParams[$k] }
      $params[$key] = $symbol
      $tries += [pscustomobject]@{ key = $key; symbol = $symbol; params = $params }
    }
  }

  foreach ($t in $tries) {
    try {
      $resp = Invoke-AkshareQuery -Url $Url -SessionId $SessionId -ApiName $ApiName -Params $t.params
      if ($resp.result.structuredContent.ok -eq $true) {
        return [pscustomobject]@{ ok = $true; api = $ApiName; used = $t; resp = $resp }
      }
    }
    catch {
      # ignore and continue
    }
  }

  return [pscustomobject]@{ ok = $false; api = $ApiName; used = $null; resp = $null }
}

function Save-AkshareStructuredContent {
  param(
    [string]$PathBase,
    [object]$StructuredContent
  )

  ($StructuredContent | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 ($PathBase + ".json")

  if ($StructuredContent.data -and ($StructuredContent.data -is [System.Collections.IEnumerable])) {
    $rows = @($StructuredContent.data)
    if ($rows.Count -gt 0 -and ($rows[0] -is [hashtable] -or $rows[0] -is [pscustomobject])) {
      try {
        $rows | Export-Csv -Encoding utf8 -NoTypeInformation ($PathBase + ".csv")
      }
      catch {
        # ignore csv export errors for odd structures
      }
    }
  }
}

Ensure-Dir $OutputDir
$stampDate = Get-Date -Format "yyyyMMdd"
$stampTs = Get-Date -Format "yyyyMMdd_HHmmss"
$ticker = ($TsCode -replace '\\.', '_')
$symbolCandidates = Convert-TsCodeToAkSymbolCandidates -Code $TsCode

$sid = Get-McpSessionId -Url $BaseUrl

# initialize
$null = Invoke-McpSseRequest -Url $BaseUrl -SessionId $sid -BodyObject @{
  jsonrpc = "2.0"
  id      = 1
  method  = "initialize"
  params  = @{
    protocolVersion = "2024-11-05"
    clientInfo      = @{ name = "codex-cli"; version = "stock-snapshot" }
    capabilities    = @{ tools = @{}; resources = @{}; prompts = @{} }
  }
}

Write-Host "Akshare MCP session: $sid"
Write-Host "Target: $TsCode"

# Discover likely APIs (best-effort) and try calls
$realtimeCandidates = @("stock_zh_a_quote_detail_em", "stock_bid_ask_em", "stock_zh_a_tick_tx")
$fundCandidates = @("stock_financial_abstract", "stock_financial_analysis_indicator", "stock_financial_report_sina")
$annCandidates = @("stock_zh_a_disclosure_report_cninfo", "stock_report_disclosure", "stock_zh_a_disclosure_relation_cninfo")
$newsCandidates = @("stock_news_em", "stock_news_main_cx")

function First-Ok {
  param(
    [string[]]$ApiNames,
    [hashtable]$BaseParams,
    [string[]]$SymbolCandidates
  )
  foreach ($api in $ApiNames) {
    $r = Try-ExecuteWithSymbolCandidates -Url $BaseUrl -SessionId $sid -ApiName $api -BaseParams $BaseParams -SymbolCandidates $SymbolCandidates
    if ($r.ok) { return $r }
  }
  return $null
}

$realtime = First-Ok -ApiNames $realtimeCandidates -BaseParams @{} -SymbolCandidates $symbolCandidates
$fund = First-Ok -ApiNames $fundCandidates -BaseParams @{} -SymbolCandidates $symbolCandidates

$endDate = Get-Date
$startDate = $endDate.AddDays(-60)
$startYmd = $startDate.ToString("yyyyMMdd")
$endYmd = $endDate.ToString("yyyyMMdd")

$anns = First-Ok -ApiNames $annCandidates -BaseParams @{ start_date = $startYmd; end_date = $endYmd } -SymbolCandidates $symbolCandidates
$news = First-Ok -ApiNames $newsCandidates -BaseParams @{} -SymbolCandidates $symbolCandidates

$bundle = [ordered]@{
  ok = $true
  generated = (Get-Date).ToString("o")
  baseUrl = $BaseUrl
  sessionId = $sid
  ts_code = $TsCode
  symbol_candidates = $symbolCandidates
  realtime = $realtime
  fundamentals = $fund
  announcements = $anns
  news = $news
}

$outBase = Join-Path $OutputDir ("akshare_{0}_{1}" -f $ticker, $stampTs)
($bundle | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 ($outBase + "_bundle.json")

if ($realtime -and $realtime.ok) {
  Save-AkshareStructuredContent -PathBase ($outBase + "_realtime") -StructuredContent $realtime.resp.result.structuredContent
  Write-Host "Realtime: OK via $($realtime.api) ($($realtime.used.key)=$($realtime.used.symbol))"
} else {
  Write-Host "Realtime: NOT FOUND (no candidate API succeeded)"
}

if ($fund -and $fund.ok) {
  Save-AkshareStructuredContent -PathBase ($outBase + "_fundamentals") -StructuredContent $fund.resp.result.structuredContent
  Write-Host "Fundamentals: OK via $($fund.api) ($($fund.used.key)=$($fund.used.symbol))"
} else {
  Write-Host "Fundamentals: NOT FOUND (no candidate API succeeded)"
}

if ($anns -and $anns.ok) {
  Save-AkshareStructuredContent -PathBase ($outBase + "_announcements") -StructuredContent $anns.resp.result.structuredContent
  Write-Host "Announcements: OK via $($anns.api) ($($anns.used.key)=$($anns.used.symbol))"
} else {
  Write-Host "Announcements: NOT FOUND (no candidate API succeeded)"
  $annSearch = @(
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "notice" -Limit 10)
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "announcement" -Limit 10)
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "disclosure" -Limit 10)
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "anns" -Limit 10)
  )
  ($annSearch | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 ($outBase + "_announcements_search.json")
  foreach ($sr in $annSearch) {
    foreach ($r in @($sr.result.structuredContent.results)) {
      Write-Host ("  candidate: {0}" -f $r.name)
    }
  }
}

if ($news -and $news.ok) {
  Save-AkshareStructuredContent -PathBase ($outBase + "_news") -StructuredContent $news.resp.result.structuredContent
  Write-Host "News: OK via $($news.api) ($($news.used.key)=$($news.used.symbol))"
} else {
  Write-Host "News: NOT FOUND (no candidate API succeeded)"
  $newsSearch = @(
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "news" -Limit 10)
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "stock_news" -Limit 10)
    (Search-AkshareDocs -Url $BaseUrl -SessionId $sid -Keyword "report" -Limit 10)
  )
  ($newsSearch | ConvertTo-Json -Depth 64) | Set-Content -Encoding utf8 ($outBase + "_news_search.json")
  foreach ($sr in $newsSearch) {
    foreach ($r in @($sr.result.structuredContent.results)) {
      Write-Host ("  candidate: {0}" -f $r.name)
    }
  }
}

Write-Host "Bundle: $($outBase)_bundle.json"
