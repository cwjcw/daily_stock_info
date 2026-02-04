# daily_stock_info

用于日线、分钟线与资金流数据的抓取、清洗与入库。

## 数据库文档

- 详见 `DB_SCHEMA.md`（各表结构、字段中文名、单位说明）

## 数据库表（统一前缀 S_）

- S_daily_raw：日线行情 + 每日指标（合并表）
- S_adj_factor：复权因子
- S_minute_5m：5分钟原始行情（不复权）
- S_moneyflow_ind：个股资金流
- S_moneyflow_sector：板块资金流（行业/概念/地域）
- S_moneyflow_mkt：大盘资金流
- S_moneyflow_hsgt：沪深港通资金流

## 单位口径（已统一）

- 历史日线、分钟线：成交量=股（字段 `vol_share`），成交额=元
- 实时数据（国金接口）：成交量=股（字段 `vol_share`），成交额=元

## 文档路径

国金接口文档路径：`E:\\Software\\stock\\gjzqqmt\\QMT操作说明文档`

## 运行示例

```powershell
python scripts/tushare_remote_scan_buy_signals.py --asof 20260130 --history-bars 10
```
