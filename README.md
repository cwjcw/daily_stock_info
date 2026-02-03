# daily_stock_info
每天收盘后复盘，以及交易时间进行判断买入卖出，基于我自己开发的MCP，结合大模型进行判断

## 扫描满足买入信号的股票（Tushare MCP）

生成结果文件：`output/tushare_buy_signals_latest.csv`

```powershell
python scripts/tushare_remote_scan_buy_signals.py --asof 20260130 --history-bars 10
```
