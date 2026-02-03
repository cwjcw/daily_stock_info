# 数据库结构说明（S_ 前缀）

> 所有 MySQL 表均以 `S_` 前缀区分。
> 字段为“全量入库”，除以下列出的核心字段外，其余字段与 Tushare/国金接口返回保持一致。

## 1. MySQL 表

### 1.1 S_daily_raw（日线行情 + 每日指标合并表）
**主键**：`(ts_code, trade_date)`

| 字段 | 中文 | 单位 | 说明 |
|---|---|---|---|
| ts_code | 股票代码 | - | 例如 600000.SH |
| trade_date | 交易日 | - | YYYYMMDD |
| open | 开盘价 | 元 | 日线开盘价 |
| high | 最高价 | 元 | 日线最高价 |
| low | 最低价 | 元 | 日线最低价 |
| close | 收盘价 | 元 | 日线收盘价 |
| pre_close | 昨收价 | 元 | 前一交易日收盘价 |
| change | 涨跌额 | 元 | close - pre_close |
| pct_chg | 涨跌幅 | % | 百分比 |
| vol | 成交量 | 手 | Tushare 原始即为手 |
| amount | 成交额 | 元 | **Tushare 原始为千元，入库前已 ×1000 转为元** |

**每日指标 (doc_id=32) 常见字段：**
turnover_rate（换手率%）、turnover_rate_f、volume_ratio（量比）、pe、pe_ttm、pb、ps、ps_ttm、dv_ratio、dv_ttm、total_share、float_share、free_share、total_mv、circ_mv 等。

---

### 1.2 S_adj_factor（复权因子）
**主键**：`(ts_code, trade_date)`

| 字段 | 中文 | 单位 | 说明 |
|---|---|---|---|
| ts_code | 股票代码 | - | |
| trade_date | 交易日 | - | YYYYMMDD |
| adj_factor | 复权因子 | - | 用于计算前复权价格 |

---

### 1.3 S_minute_raw（1分钟原始行情，不复权）
**主键**：`(ts_code, trade_time)`

| 字段 | 中文 | 单位 | 说明 |
|---|---|---|---|
| ts_code | 股票代码 | - | |
| trade_time | 交易时间 | - | YYYYMMDDHHMMSS |
| trade_date | 交易日 | - | YYYYMMDD |
| time | 时间戳 | - | 与 trade_time 相同格式 |
| open/high/low/close | 价格 | 元 | 分钟线价格 |
| vol | 成交量 | 手 | 国金接口原始为手（不换算） |
| amount | 成交额 | 元 | 国金接口原始为元 |

> 其余字段按国金接口原样入库。

---

### 1.4 S_moneyflow_ind（个股资金流，doc_id=349）
**主键**：`(ts_code, trade_date)`

常见字段：
- trade_date（交易日）
- ts_code（股票代码）
- name（名称）
- pct_change（涨跌幅%）
- close（收盘价）
- net_amount（净流入额）
- buy_elg_amount/buy_lg_amount/buy_md_amount/buy_sm_amount（超大/大/中/小单流入）
- *_rate（对应占比%）

**单位**：金额类字段单位为 **万元**（Tushare moneyflow_dc 口径）。

---

### 1.5 S_moneyflow_sector（板块资金流，doc_id=344）
**主键**：`(ts_code, trade_date, content_type)`

常见字段：
- trade_date（交易日）
- ts_code（板块代码）
- name（板块名称）
- content_type（行业/概念/地域）
- net_amount 等资金流字段

**单位**：金额类字段单位为 **万元**（Tushare moneyflow_ind_dc 口径）。

---

### 1.6 S_moneyflow_mkt（大盘资金流，doc_id=345）
**主键**：`(trade_date)`

常见字段：
- trade_date（交易日）
- net_amount / buy_amount / sell_amount 等

**单位**：金额类字段单位为 **万元**（Tushare moneyflow_mkt_dc 口径）。

---

### 1.7 S_moneyflow_hsgt（沪深港通资金流，doc_id=47）
**主键**：`(trade_date)`

常见字段：
- trade_date（交易日）
- ggt_ss / ggt_sz（港股通）
- hgt / sgt（沪/深股通）
- north_money / south_money（北向/南向资金）

**单位**：金额类字段单位为 **万元**（Tushare 口径）。

---

## 2. SQLite 实时缓存（temp_cache/realtime_cache.db）

### S_realtime_quote（实时行情缓存，国金接口）
**主键**：`(ts_code, time)`

| 字段 | 中文 | 单位 | 说明 |
|---|---|---|---|
| ts_code | 股票代码 | - | |
| time | 时间戳 | - | YYYYMMDDHHMMSS |
| price | 现价 | 元 | 实时价 |
| open/high/low | 价格 | 元 | |
| pre_close | 昨收 | 元 | |
| vol | 成交量 | 手 | 国金实时口径（手） |
| amount | 成交额 | 元 | 国金实时口径（元） |
| source | 数据来源 | - | 例如 xtquant |
| raw_json | 原始数据 | - | 原始字段 JSON 备份 |

---

## 3. 参考文档

- Tushare 日线（doc_id=27）
- Tushare 每日指标（doc_id=32）
- Tushare 复权因子（doc_id=28）
- Tushare 资金流：doc_id=349/344/345/47
- 国金接口文档：`E:\\Software\\stock\\gjzqqmt\\QMT操作说明文档`
