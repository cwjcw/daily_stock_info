#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd
import requests
import subprocess
import threading
import time
import queue


class McpError(RuntimeError):
    pass


@dataclass(frozen=True)
class McpClientInfo:
    name: str = "daily_stock_info"
    version: str = "tushare_scan_buy_signals"


class HttpMcpClient:
    def __init__(self, base_url: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session_id = self._get_session_id()
        self._initialize()

    def _get_session_id(self) -> str:
        resp = self.session.get(
            self.base_url,
            headers={"Accept": "text/event-stream"},
            timeout=self.timeout_s,
        )
        sid = resp.headers.get("mcp-session-id")
        if not sid:
            raise McpError(f"'mcp-session-id' header not found from {self.base_url}")
        return sid.strip()

    def _post_jsonrpc(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(
            self.base_url,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Mcp-Session-Id": self.session_id,
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()

        text = resp.text
        data_lines = [line for line in text.splitlines() if re.match(r"^\s*data:\s*", line)]
        if not data_lines:
            raise McpError(f"No 'data:' line found in response. Raw response:\n{text}")
        payload = re.sub(r"^\s*data:\s*", "", data_lines[-1]).strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            raise McpError(f"Failed to parse JSON payload from SSE. payload={payload!r}") from e

    def _initialize(self) -> None:
        init_resp = self._post_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {
                        "name": McpClientInfo.name,
                        "version": McpClientInfo.version,
                    },
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                },
            }
        )
        if "error" in init_resp:
            raise McpError(f"MCP initialize failed: {init_resp['error']}")

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self._post_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        if "error" in resp:
            raise McpError(f"tools/list failed: {resp['error']}")
        tools = resp.get("result", {}).get("tools", [])
        if not isinstance(tools, list):
            raise McpError(f"Unexpected tools/list response: {resp}")
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resp = self._post_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if "error" in resp:
            raise McpError(f"tools/call failed: {resp['error']}")
        return resp


class StdioMcpClient:
    """
    Minimal MCP (JSON-RPC) client over stdio using newline-delimited JSON.

    This matches FastMCP's stdio transport: each message is a single JSON line.
    """

    def __init__(self, command: list[str], timeout_s: int = 180) -> None:
        self.command = command
        self.timeout_s = timeout_s
        self._next_id = 1
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self._proc.stdin and self._proc.stdout and self._proc.stderr
        self._stderr_buf: list[bytes] = []
        self._stdout_q: "queue.Queue[bytes]" = queue.Queue()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread.start()

        init_id = self._next_id
        self._next_id += 1
        init_resp = self._request(
            {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": McpClientInfo.name, "version": McpClientInfo.version},
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                },
            },
            wait_id=init_id,
        )
        if "error" in init_resp:
            raise McpError(f"MCP initialize failed: {init_resp['error']}")
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _drain_stdout(self) -> None:
        assert self._proc.stdout
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            self._stdout_q.put(line)

    def _drain_stderr(self) -> None:
        assert self._proc.stderr
        while True:
            chunk = self._proc.stderr.read(4096)
            if not chunk:
                break
            self._stderr_buf.append(chunk)

    def _write(self, message: dict[str, Any]) -> None:
        if self._proc.poll() is not None:
            stderr = b"".join(self._stderr_buf).decode("utf-8", errors="replace")
            raise McpError(f"MCP server exited early. stderr:\n{stderr}")
        assert self._proc.stdin
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self._proc.stdin.write(payload + b"\n")
        self._proc.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        try:
            line = self._stdout_q.get(timeout=self.timeout_s)
        except queue.Empty as e:
            raise McpError("Timed out waiting for MCP response line") from e
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise McpError(f"Failed to parse MCP JSON line. line={line!r}") from e

    def _request(self, message: dict[str, Any], wait_id: int) -> dict[str, Any]:
        self._write(message)
        while True:
            resp = self._read_message()
            if resp.get("id") == wait_id:
                return resp

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def list_tools(self) -> list[dict[str, Any]]:
        req_id = self._next_id
        self._next_id += 1
        resp = self._request({"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}, wait_id=req_id)
        if "error" in resp:
            raise McpError(f"tools/list failed: {resp['error']}")
        tools = resp.get("result", {}).get("tools", [])
        if not isinstance(tools, list):
            raise McpError(f"Unexpected tools/list response: {resp}")
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        resp = self._request(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            wait_id=req_id,
        )
        if "error" in resp:
            raise McpError(f"tools/call failed: {resp['error']}")
        return resp


class McpClient(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _structured_to_df(tool_call_resp: dict[str, Any]) -> pd.DataFrame:
    payload = _tool_result_payload(tool_call_resp)
    data = payload.get("data")
    meta = payload.get("meta") or {}
    columns = meta.get("columns") or payload.get("columns") or payload.get("fields")

    if isinstance(data, list) and (not data):
        return pd.DataFrame(columns=columns or None)

    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame.from_records(data)

    if isinstance(data, list) and data and isinstance(data[0], list) and isinstance(columns, list):
        return pd.DataFrame(data, columns=columns)

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return pd.DataFrame.from_records(data["records"])

    raise McpError(f"Unrecognized tool payload format: keys={list(payload.keys())}")


def _tool_result_payload(tool_call_resp: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize MCP tool response to a dict payload containing at least:
      - data: list[dict] or list[list]
      - meta: dict (optional)

    This supports both:
      - result.structuredContent (common)
      - result.content[0].text containing a JSON string (FastMCP stdio)
    """
    result = tool_call_resp.get("result", {}) or {}
    if "structuredContent" in result and result["structuredContent"] is not None:
        sc = result.get("structuredContent") or {}
        return sc if isinstance(sc, dict) else {"data": sc}

    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0] or {}
        if isinstance(first, dict) and first.get("type") == "text":
            text = first.get("text", "")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                raise McpError("Failed to parse JSON text payload from tool response") from e
            if not isinstance(payload, dict):
                raise McpError(f"Unexpected tool payload type: {type(payload)}")
            return payload

    raise McpError(f"Unrecognized tool response shape: keys={list(result.keys())}")


def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _get_last_open_trade_date(client: McpClient, today: str) -> str:
    start = _yyyymmdd(datetime.strptime(today, "%Y%m%d") - timedelta(days=200))
    resp = client.call_tool(
        "execute_tushare_query",
        {
            "api_name": "trade_cal",
            "params": {
                "exchange": "SSE",
                "start_date": start,
                "end_date": today,
                "is_open": "1",
                "fields": "cal_date,is_open",
            },
        },
    )
    cal = _structured_to_df(resp)
    if cal.empty:
        raise McpError("trade_cal returned empty result; cannot determine last trade date")
    cal = cal[cal["is_open"].astype(int) == 1].sort_values("cal_date")
    return str(cal["cal_date"].iloc[-1])


def _get_recent_trade_dates(client: McpClient, end_date: str, bars: int) -> list[str]:
    start = _yyyymmdd(datetime.strptime(end_date, "%Y%m%d") - timedelta(days=400))
    resp = client.call_tool(
        "execute_tushare_query",
        {
            "api_name": "trade_cal",
            "params": {
                "exchange": "SSE",
                "start_date": start,
                "end_date": end_date,
                "is_open": "1",
                "fields": "cal_date,is_open",
            },
        },
    )
    cal = _structured_to_df(resp)
    cal = cal[cal["is_open"].astype(int) == 1].sort_values("cal_date")
    dates = [str(x) for x in cal["cal_date"].tolist()]
    if len(dates) < bars:
        return dates
    return dates[-bars:]


def _fetch_stock_basic(client: McpClient) -> pd.DataFrame:
    df = _fetch_all_pages(
        client,
        api_name="stock_basic",
        params={"list_status": "L", "fields": "ts_code,name,market,list_status"},
        label="stock_basic",
    )
    if df.empty:
        raise McpError("stock_basic returned empty result")
    return df


def _fetch_all_pages(client: McpClient, api_name: str, params: dict[str, Any], label: str) -> pd.DataFrame:
    resp0 = client.call_tool("execute_tushare_query", {"api_name": api_name, "params": params})
    payload0 = _tool_result_payload(resp0)
    meta0 = payload0.get("meta") or {}
    df0 = _structured_to_df(resp0)
    if df0.empty:
        return df0

    truncated = bool(meta0.get("truncated"))
    total_rows = int(meta0.get("total_rows") or len(df0))
    max_rows = int(meta0.get("max_rows") or len(df0) or 100)

    frames: list[pd.DataFrame] = [df0]
    print(f"Fetched {label} rows: {len(df0)} (offset=0)", file=sys.stderr)
    if not truncated or total_rows <= len(df0):
        return df0

    for offset in range(max_rows, total_rows, max_rows):
        page_params = dict(params)
        page_params["offset"] = offset
        page_params["limit"] = max_rows
        resp = client.call_tool("execute_tushare_query", {"api_name": api_name, "params": page_params})
        df = _structured_to_df(resp)
        if df.empty:
            break
        frames.append(df)
        print(f"Fetched {label} rows: {len(df)} (offset={offset})", file=sys.stderr)
    return pd.concat(frames, ignore_index=True)


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def _calc_kdj_kd(rsv: pd.Series) -> pd.DataFrame:
    arr = rsv.to_numpy(dtype=float)
    k = np.empty_like(arr)
    d = np.empty_like(arr)
    k_prev = 50.0
    d_prev = 50.0
    for i in range(len(arr)):
        val = arr[i]
        if not np.isfinite(val):
            val = 0.0
        k_prev = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * val
        d_prev = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_prev
        k[i] = k_prev
        d[i] = d_prev
    return pd.DataFrame({"kdj_k": k, "kdj_d": d}, index=rsv.index)


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"], ascending=[True, True])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("ts_code", group_keys=False)

    df["ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    df["ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["boll_mid"] = df["ma20"]

    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["dif"] = ema12 - ema26
    df["dea"] = g["dif"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["dif"] - df["dea"]

    df["rsi6"] = g["close"].transform(lambda s: _calc_rsi(s, 6))
    df["rsi12"] = g["close"].transform(lambda s: _calc_rsi(s, 12))

    low9 = g["low"].transform(lambda s: s.rolling(9).min())
    high9 = g["high"].transform(lambda s: s.rolling(9).max())
    denom = (high9 - low9).replace(0, np.nan)
    df["rsv"] = ((df["close"] - low9) / denom * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    kd = g["rsv"].apply(_calc_kdj_kd)
    df = df.join(kd)

    return df


def _evaluate_buy_signals(latest: pd.DataFrame) -> pd.DataFrame:
    cond = (
        (latest["ma5"] > latest["ma10"])
        & (latest["ma10"] > latest["ma20"])
        & (latest["close"] > latest["ma20"])
        & (latest["dif"] > latest["dea"])
        & (latest["macd_hist"] > 0)
        & (latest["rsi6"] > 50)
        & (latest["rsi6"] > latest["rsi12"])
        & (latest["kdj_k"] > latest["kdj_d"])
        & (latest["kdj_k"] < 80)
        & (latest["close"] > latest["boll_mid"])
    )
    return latest[cond].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan A-shares that meet all buy signals (MA/MACD/RSI/KDJ/BOLL).")
    parser.add_argument(
        "--transport",
        default=os.environ.get("MCP_TRANSPORT", "stdio-ssh"),
        choices=["http", "stdio-ssh"],
        help="MCP transport: http (SSE JSON-RPC) or stdio-ssh (ssh -> docker run tushare-mcp).",
    )
    parser.add_argument("--base-url", default=os.environ.get("MCP_BASE_URL", "http://10.147.20.211:1818/mcp"))
    parser.add_argument("--ssh-target", default=os.environ.get("TUSHARE_MCP_SSH_TARGET", "cwj@10.147.20.211"))
    parser.add_argument(
        "--remote-cmd",
        default=os.environ.get(
            "TUSHARE_MCP_REMOTE_CMD",
            "docker run --rm -i --env-file /home/cwj/code/TushareMCP/.env -v /home/cwj/code/TushareMCP/data:/app/data tushare-mcp",
        ),
        help="Remote command to start the MCP server (stdio).",
    )
    parser.add_argument("--asof", default=os.environ.get("ASOF", datetime.now().strftime("%Y%m%d")))
    parser.add_argument(
        "--history-bars",
        type=int,
        default=int(os.environ.get("HISTORY_BARS", "10")),
        help="How many recent trading days to pull for MA5/MA10 calculation. Minimum effective is 10.",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--latest-name", default="tushare_buy_signals_latest.csv")
    parser.add_argument("--timeout-s", type=int, default=int(os.environ.get("MCP_TIMEOUT_S", "180")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client: Any
    if args.transport == "http":
        client = HttpMcpClient(args.base_url, timeout_s=args.timeout_s)
    else:
        client = StdioMcpClient(
            command=[
                "ssh",
                "-o",
                "BatchMode=yes",
                "-T",
                args.ssh_target,
                args.remote_cmd,
            ],
            timeout_s=args.timeout_s,
        )

    tools = {t.get("name") for t in client.list_tools()}
    if "execute_tushare_query" not in tools:
        raise McpError(f"Tool 'execute_tushare_query' not found. Available sample tools: {sorted([x for x in tools if x])[:20]}")

    asof_trade_date = _get_last_open_trade_date(client, args.asof)
    history_bars = max(int(args.history_bars), 10)
    trade_dates = _get_recent_trade_dates(client, asof_trade_date, history_bars)
    if not trade_dates or len(trade_dates) < 10:
        raise McpError("No trading dates returned from trade_cal")
    prior_dates = trade_dates[:-1]

    stock_basic = _fetch_stock_basic(client)[["ts_code", "name"]]

    factor = _fetch_all_pages(
        client,
        api_name="stk_factor",
        params={"trade_date": asof_trade_date},
        label=f"stk_factor({asof_trade_date})",
    )
    if factor.empty:
        raise McpError(f"stk_factor returned empty result for trade_date={asof_trade_date}")
    factor["trade_date"] = factor["trade_date"].astype(str)
    factor = factor.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    daily_frames: list[pd.DataFrame] = []
    for d in prior_dates:
        day = _fetch_all_pages(
            client,
            api_name="daily",
            params={"trade_date": d, "fields": "ts_code,trade_date,close"},
            label=f"daily({d})",
        )
        if not day.empty:
            daily_frames.append(day[["ts_code", "trade_date", "close"]])

    closes = pd.concat(daily_frames + [factor[["ts_code", "trade_date", "close"]]], ignore_index=True)
    closes["trade_date"] = closes["trade_date"].astype(str)
    closes["close"] = pd.to_numeric(closes["close"], errors="coerce")
    closes = closes.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    closes = closes.sort_values(["ts_code", "trade_date"], ascending=[True, True])
    g = closes.groupby("ts_code", group_keys=False)
    closes["ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    closes["ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())

    latest_ma = (
        closes[closes["trade_date"] == asof_trade_date][["ts_code", "ma5", "ma10"]]
        .sort_values(["ts_code"])
        .drop_duplicates(subset=["ts_code"], keep="last")
    )

    latest = factor.merge(latest_ma, on="ts_code", how="left")
    latest = latest.merge(stock_basic.drop_duplicates(subset=["ts_code"], keep="last"), on="ts_code", how="left")

    latest["ma20"] = pd.to_numeric(latest.get("boll_mid"), errors="coerce")
    latest["dif"] = pd.to_numeric(latest.get("macd_dif"), errors="coerce")
    latest["dea"] = pd.to_numeric(latest.get("macd_dea"), errors="coerce")
    latest["macd_hist"] = pd.to_numeric(latest.get("macd"), errors="coerce")
    latest["rsi6"] = pd.to_numeric(latest.get("rsi_6"), errors="coerce")
    latest["rsi12"] = pd.to_numeric(latest.get("rsi_12"), errors="coerce")
    latest["kdj_k"] = pd.to_numeric(latest.get("kdj_k"), errors="coerce")
    latest["kdj_d"] = pd.to_numeric(latest.get("kdj_d"), errors="coerce")
    latest["boll_mid"] = pd.to_numeric(latest.get("boll_mid"), errors="coerce")
    latest["close"] = pd.to_numeric(latest.get("close"), errors="coerce")

    hits = _evaluate_buy_signals(latest).sort_values(["ts_code"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output_name.strip() or f"tushare_buy_signals_{asof_trade_date}_{stamp}.csv"
    out_path = out_dir / out_name
    latest_path = out_dir / args.latest_name

    columns = [
        "ts_code",
        "name",
        "trade_date",
        "close",
        "ma5",
        "ma10",
        "ma20",
        "dif",
        "dea",
        "macd_hist",
        "rsi6",
        "rsi12",
        "kdj_k",
        "kdj_d",
        "boll_mid",
    ]
    for c in columns:
        if c not in hits.columns:
            hits[c] = np.nan
    hits = hits[columns]

    hits.to_csv(out_path, index=False, encoding="utf-8-sig")
    hits.to_csv(latest_path, index=False, encoding="utf-8-sig")

    meta = {
        "ok": True,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "asof_input": args.asof,
        "asof_trade_date": asof_trade_date,
        "history_bars": history_bars,
        "daily_rows": int(len(closes)),
        "factor_rows": int(len(factor)),
        "matched": int(len(hits)),
        "output": str(out_path),
        "latest": str(latest_path),
    }
    (out_dir / "tushare_buy_signals_meta_latest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except McpError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
