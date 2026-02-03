#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


REMOTE_PY = r"""
import os, json
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

asof = os.environ.get("ASOF") or datetime.now().strftime("%Y%m%d")
history_bars = max(int(os.environ.get("HISTORY_BARS", "10")), 10)

token = os.environ.get("TUSHARE_TOKEN")
if not token:
    raise SystemExit("Missing TUSHARE_TOKEN in environment")

ts.set_token(token)
pro = ts.pro_api()

start = (datetime.strptime(asof, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=asof, is_open="1")
if cal is None or cal.empty:
    raise SystemExit("trade_cal empty")
cal = cal.sort_values("cal_date")
asof_trade_date = str(cal["cal_date"].iloc[-1])
dates = [str(x) for x in cal["cal_date"].tolist()][-history_bars:]
prior_dates = dates[:-1]

frames = []
for d in prior_dates:
    day = pro.daily(trade_date=d)
    if day is None or day.empty:
        continue
    frames.append(day[["ts_code", "trade_date", "close"]])

factor = pro.stk_factor(trade_date=asof_trade_date)
if factor is None or factor.empty:
    raise SystemExit(f"stk_factor empty for {asof_trade_date}")

factor_cols = [
    "ts_code",
    "trade_date",
    "close",
    "macd_dif",
    "macd_dea",
    "macd",
    "rsi_6",
    "rsi_12",
    "kdj_k",
    "kdj_d",
    "boll_mid",
]
missing = [c for c in factor_cols if c not in factor.columns]
if missing:
    raise SystemExit(f"stk_factor missing columns: {missing}")
factor = factor[factor_cols]

closes = pd.concat(frames + [factor[["ts_code", "trade_date", "close"]]], ignore_index=True)
closes["trade_date"] = closes["trade_date"].astype(str)
closes["close"] = pd.to_numeric(closes["close"], errors="coerce")
closes = closes.drop_duplicates(["ts_code", "trade_date"], keep="last").sort_values(["ts_code", "trade_date"])

g = closes.groupby("ts_code", group_keys=False)
closes["ma5"] = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
closes["ma10"] = g["close"].rolling(10).mean().reset_index(level=0, drop=True)
latest_ma = (
    closes[closes["trade_date"] == asof_trade_date][["ts_code", "ma5", "ma10"]]
    .drop_duplicates("ts_code", keep="last")
)

basic = pro.stock_basic(list_status="L", fields="ts_code,name")
if basic is None or basic.empty:
    basic = pd.DataFrame(columns=["ts_code", "name"])

latest = factor.merge(latest_ma, on="ts_code", how="left").merge(basic, on="ts_code", how="left")

latest["ma20"] = pd.to_numeric(latest["boll_mid"], errors="coerce")
latest["dif"] = pd.to_numeric(latest["macd_dif"], errors="coerce")
latest["dea"] = pd.to_numeric(latest["macd_dea"], errors="coerce")
latest["macd_hist"] = pd.to_numeric(latest["macd"], errors="coerce")
latest["rsi6"] = pd.to_numeric(latest["rsi_6"], errors="coerce")
latest["rsi12"] = pd.to_numeric(latest["rsi_12"], errors="coerce")
latest["kdj_k"] = pd.to_numeric(latest["kdj_k"], errors="coerce")
latest["kdj_d"] = pd.to_numeric(latest["kdj_d"], errors="coerce")
latest["boll_mid"] = pd.to_numeric(latest["boll_mid"], errors="coerce")
latest["close"] = pd.to_numeric(latest["close"], errors="coerce")
latest["trade_date"] = latest["trade_date"].astype(str)

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

hits = latest[cond].copy()
hits = hits.drop_duplicates("ts_code", keep="last").sort_values("ts_code")

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
hits = hits[columns]

out_dir = "/app/data"
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(out_dir, f"tushare_buy_signals_{asof_trade_date}_{stamp}.csv")
latest_path = os.path.join(out_dir, "tushare_buy_signals_latest.csv")
meta_path = os.path.join(out_dir, "tushare_buy_signals_meta_latest.json")

hits.to_csv(out_path, index=False, encoding="utf-8-sig")
hits.to_csv(latest_path, index=False, encoding="utf-8-sig")

meta = {
    "ok": True,
    "generated": datetime.now().isoformat(timespec="seconds"),
    "asof_input": asof,
    "asof_trade_date": asof_trade_date,
    "history_bars": history_bars,
    "daily_rows": int(len(closes)),
    "factor_rows": int(len(factor)),
    "matched": int(len(hits)),
    "output": out_path,
    "latest": latest_path,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(meta, ensure_ascii=False))
"""


def run(cmd: list[str], *, input_text: str | None = None, timeout_s: int = 1200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run buy-signal scan via remote docker (uses Tushare token from env-file).")
    parser.add_argument("--ssh-target", default=os.environ.get("TUSHARE_MCP_SSH_TARGET", "cwj@10.147.20.211"))
    parser.add_argument("--remote-env-file", default="/home/cwj/code/TushareMCP/.env")
    parser.add_argument("--remote-data-dir", default="/home/cwj/code/TushareMCP/data")
    parser.add_argument("--docker-image", default="tushare-mcp")
    parser.add_argument("--asof", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--history-bars", type=int, default=10)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--timeout-s", type=int, default=1200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docker_cmd = (
        f"docker run --rm -i --env-file {args.remote_env_file} "
        f"-v {args.remote_data_dir}:/app/data "
        f"-e ASOF={args.asof} -e HISTORY_BARS={max(args.history_bars, 10)} "
        f"{args.docker_image} python -"
    )

    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-T", args.ssh_target, docker_cmd]
    proc = run(ssh_cmd, input_text=REMOTE_PY, timeout_s=args.timeout_s)

    # The remote script prints exactly one JSON line.
    meta = json.loads(proc.stdout.strip().splitlines()[-1])
    remote_latest = f"{args.remote_data_dir}/tushare_buy_signals_latest.csv"
    remote_meta = f"{args.remote_data_dir}/tushare_buy_signals_meta_latest.json"
    remote_out = meta.get("output", "")
    remote_out_basename = os.path.basename(remote_out) if remote_out else ""

    run(["scp", "-o", "BatchMode=yes", f"{args.ssh_target}:{remote_latest}", str(out_dir / "tushare_buy_signals_latest.csv")], timeout_s=args.timeout_s)
    run(["scp", "-o", "BatchMode=yes", f"{args.ssh_target}:{remote_meta}", str(out_dir / "tushare_buy_signals_meta_latest.json")], timeout_s=args.timeout_s)
    if remote_out_basename:
        run(
            ["scp", "-o", "BatchMode=yes", f"{args.ssh_target}:{args.remote_data_dir}/{remote_out_basename}", str(out_dir / remote_out_basename)],
            timeout_s=args.timeout_s,
        )

    print(json.dumps({**meta, "local_latest": str(out_dir / 'tushare_buy_signals_latest.csv')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
