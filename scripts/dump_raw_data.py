# -*- coding: utf-8 -*-
"""
Dump raw market data to CSV for debugging:
- last 200 trading days of 1d bars
- last 5 trading days of 1m bars (approx 5*240)
Outputs to temp_cache/raw_dump_YYYYMMDD/
"""

import datetime
import os
from typing import Dict, Any

import pandas as pd
from xtquant import xtdata

from config import CONFIG


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    port = CONFIG["PORT"]
    scan_sector = CONFIG["SCAN_SECTOR"]
    min_k_bars = CONFIG["MIN_K_BARS"]
    minute_count = 5 * 240
    dividend_type = CONFIG["DIVIDEND_TYPE"]
    cache_dir = CONFIG["CACHE_DIR"]

    ts = datetime.datetime.now().strftime("%Y%m%d")
    base_dir = os.path.join(cache_dir, f"raw_dump_{ts}")
    dir_1d = os.path.join(base_dir, "1d")
    dir_1m = os.path.join(base_dir, "1m")
    ensure_dir(dir_1d)
    ensure_dir(dir_1m)

    print(f"📡 Connecting xtdata (Port: {port})...")
    xtdata.connect(port=port)

    stock_list = xtdata.get_stock_list_in_sector(scan_sector)
    filtered_list = [s for s in stock_list if not s.startswith(("688", "8", "4"))]

    print(f"📥 Fetching data for {len(filtered_list)} symbols...")
    data_1d: Dict[str, pd.DataFrame] = xtdata.get_market_data_ex(
        [], filtered_list, period="1d", count=min_k_bars, dividend_type=dividend_type
    )
    data_1m: Dict[str, pd.DataFrame] = xtdata.get_market_data_ex(
        [], filtered_list, period="1m", count=minute_count, dividend_type=dividend_type
    )

    total = len(filtered_list)
    for idx, code in enumerate(filtered_list, start=1):
        if idx % 200 == 0 or idx == total:
            print(f"⏳ Progress: {idx}/{total}")
        df_1d = data_1d.get(code)
        df_1m = data_1m.get(code)
        if isinstance(df_1d, pd.DataFrame) and not df_1d.empty:
            df_1d.to_csv(os.path.join(dir_1d, f"{code}_1d.csv"))
        if isinstance(df_1m, pd.DataFrame) and not df_1m.empty:
            df_1m.to_csv(os.path.join(dir_1m, f"{code}_1m.csv"))

    print(f"✅ Dump complete: {base_dir}")


if __name__ == "__main__":
    main()
