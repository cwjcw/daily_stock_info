# 进阶量化交易规则（整理版 MD）

## 0. 全局环境风控（总控开关）

在计算个股得分前，先判断大盘（上证指数 `000001.SH`）环境，决定整体仓位上限与策略系数。

| 大盘状态 | 判定条件 | 策略调整系数 |
|---|---|---|
| 强势市场 | Index > MA20 且 MA20 向上 | 1.0（正常开仓/重仓） |
| 震荡市场 | Index 围绕 MA20 震荡 或 Slope 平缓 | 0.8（标准仓位，提高分数门槛） |
| 弱势市场 | Index < MA20 且 MA20 向下 | 0.0（空仓/熔断，停止所有买入） |

---

## 1. 评分与执行总框架

### 1.1 维度权重分配

| 维度 | 权重 (w) | 核心逻辑 |
|---|---:|---|
| 趋势 (Trend) | 0.30 | 均线方向与斜率（底线） |
| 动能 (MACD) | 0.20 | 趋势的加速度 |
| 强度 (RSI) | 0.20 | 多周期共振与内力 |
| 摆动 (KDJ) | 0.15 | 拐点与背离（精准择时） |
| 通道 (BOLL) | 0.10 | 相对位置与开口 |
| 能量 (VR) | 0.05 | 资金确认 |

**总分计算：**  
`Score_Total = Σ (Score_i * w_i)`，满分 1.0。

### 1.2 否决机制（Veto System）

- **Hard Veto（硬否决）**：触发生死红线 → `Score = 0`，强制不买 / 清仓  
- **Soft Veto（软否决）**：触发瑕疵条件 → `Score` 上限锁定为 `0.4`，禁止开新仓；持仓者减仓或观望

### 1.3 数据精度与递归收敛（Hard Fail）

下列校验不通过时，**当日该标的直接剔除，不参与评分/排序**（优于“打 0 分”）：

- **校验 A：OHLC 合法性**
  - `Low <= min(Open, Close) <= High`
  - `High >= Low`
  - `Volume >= 0`
  - 缺失值：最近 `MIN_K_BARS` 内，任意关键字段（O/H/L/C/Vol）缺失占比 > `MAX_MISSING_RATIO` → Hard Fail
- **校验 B：复权异常跳变**
  - 前复权日线收益：`ret = Ct/Ct-1 - 1`
  - 若 `abs(ret) > RET_ABS_MAX` 且**非涨跌停** → Hard Fail
  - 若可取“除权除息/拆分合并事件日”，事件日可豁免，但要求事件日前后收益序列平滑（事件日后 3 天 abs(ret) 不连续超阈值）
- **校验 C：递归指标收敛窗口**
  - RSI/KDJ/MACD 递归计算要求：有效连续日线长度 `>= MIN_K_BARS`
  - 若存在长停牌/缺口：`MAX_GAP_DAYS` 以上断裂 → Hard Fail（不做“分段重启”）

**工程实现中的 Hard Fail 原因标签（日志用）：**
- `missing_ratio`：关键字段缺失占比超阈值
- `gap_days`：日线序列断裂超过 `MAX_GAP_DAYS`
- `insufficient_bars`：有效日线长度不足 `MIN_K_BARS`
- `ohlc_or_volume`：OHLC 或成交量合法性不通过
- `ret_jump`：复权收益异常跳变且非涨跌停

---

## 2. 买入评分细则（标准化表）

| 维度 | 指标 | 满分 (1.0) 条件 | 建议分 (0.6 - 0.8) 条件 | 否决 (Veto) |
|---|---|---|---|---|
| 趋势 | MA20 + Slope | Ct > MA20 且 Slope20 > 0.10% 且 Dev20 ≤ 6% | **0.6**：Ct > MA20 且 Slope20 走平 (0~0.1%)；**0.8**：趋势强但乖离率大 Dev20 ∈ (6%, 10%] | **Hard**：Ct < MA20；**Soft**：Slope20 < 0（若有 KDJ 底背离，则豁免：Hard → Soft） |
| 动能 | MACD | DIF > DEA 且红柱增长：HIST(t) > HIST(t-1) | **0.6**：DIF > DEA 但红柱缩短；**0.8**：DIF 在 0 轴附近刚刚金叉 | **Soft**：DIF < DEA（死叉）；**Hard**：DIF < 0 且死叉强 |
| 强度 | RSI (6/12/24) | 三线共振：RSI6 > RSI12 > RSI24 且均 > 50；趋势向上：RSI6(t) > RSI6(t-2)（容忍单日波动） | **0.6**：低位金叉（RSI6 < 35 上穿 RSI12）；**0.8**：RSI6、RSI12 > 50 且 RSI6 创新高 | **Hard**：RSI6 < 40；**Soft**：RSI6 > 75 且掉头向下 |
| 摆动 | KDJ（J 线） | 底背离（见细则）或低位 V 型反转 | **0.6**：J 线单边上行但未背离；**0.8**：J 线低位 V 反但 K < D | **Hard**：J 线向下勾头（J(t) < J(t-1)）；**Hard**：J > 100 |
| 通道 | BOLL | Ct > MB 且开口扩张：BW(t) > BW(t-1) 且 BW(t) > Mean(BW, 10) | **0.6**：Ct > MB 但通道收口；**0.8**：回踩确认（Lt 触碰中轨后回升） | **Hard**：Ct < MB（跌破中轨） |
| 能量 | VR（量比） | VR ∈ [1.5, 3.5] 且 Ct > Open（早盘 10:00 前放宽至 [1.5, 5.0]） | **0.6**：VR ∈ [1.0, 1.5)（温和放量）；**0.8**：VR ∈ [1.5, 3.5] 但收假阴线 | **Soft**：VR < 0.6 或 VR > 6.0；**早盘**（09:30-10:00）改为 VR < 0.4 或 VR > 8.0，且需连续 2 次采样触发 |

---

## 3. 核心进阶逻辑定义（Python 逻辑）

### 3.1 KDJ 进阶逻辑（底背离与拐点）

定义：
- `L5 = min(Low, window=5)`
- `J_min5 = min(J, window=5)`

**判定 1：经典底背离（Score = 1.0）**
- 条件 A：股价创 5 日新低（`Low(t) == L5` 或 `Close(t)` 接近 `L5`）
- 条件 B：J 线不创 5 日新低（`J(t) > J_min5` 且 `J_min5` 出现在 `t-2` 或 `t-3`）
- 条件 C：J 线勾头（`J(t) > J(t-1)`）

**判定 2：V 型反转（Score = 1.0）**
- 条件：`J(t-1) < 15`（超跌） AND `J(t) > J(t-1)` AND `J(t-1) < J(t-2)`

### 3.2 Slope20 的“背离豁免”机制

**解决痛点：** 防止 V 型反转第一根阳线因为均线还未拐头而被误杀。

- 默认：若 `Slope20 < 0` → Soft Veto（限制得分）
- 豁免特例：若 `Slope20 < 0` BUT (`Score_KDJ == 1.0` OR `Score_MACD == 1.0`)
- 结果：解除 Veto，允许以 `0.6 ~ 0.7` 的总分试探性开仓

### 3.3 BOLL 带宽（BW）的趋势判定

**解决痛点：** 避免因单日波动导致“连续3日张口”条件过于严苛。

- 公式：`BW = (Upper - Lower) / Middle`
- 判定：`BW(t) > BW(t-1)` AND `BW(t) > RollingMean(BW, 10)`
- 含义：只要今日带宽比昨日大，且处于近期平均水平之上，即视为“扩张趋势”

---

## 4. 卖出评分细则（风险控制）

**卖出原则：** 一旦触发 Hard Veto，无视总分直接清仓。

| 维度 | 指标 | 满分卖出 (1.0) / Hard Veto（清仓） | 建议减仓 (0.6 - 0.8) |
|---|---|---|---|
| 趋势 | MA20 | **Hard Veto**：Ct < MA20（有效跌破：幅度 > 1% 或连续 2 日） | **0.8**：Slope20 拐头向下 且 Ct 贴近均线 |
| 排列 | MA 组合 | **Hard Veto**：MA5 < MA20（短期趋势崩坏） | **0.6**：MA5 < MA10（短期死叉） |
| 动能 | MACD | **Hard Veto**：DIF < 0（进入空头市场） | **0.8**：DIF 高位死叉（DIF > 0 但下穿 DEA） |
| 摆动 | KDJ | **Hard Veto**：顶背离（价创新高、J 不创新高 + J 掉头） | **0.6**：J > 100 钝化后首根阴线 |
| 强度 | RSI | **Hard Veto**：RSI6 < 40（动能丧失） | **0.8**：RSI6 > 80 且下穿 RSI12 |
| 通道 | BOLL | **Hard Veto**：Ct < LowerBand（跌出下轨） | **0.6**：触碰上轨受阻回落 |

---

## 5. 执行策略（Action Plan）

### 5.1 信号生成步骤

1) 数据清洗：获取最近 200 日数据，执行前复权  
2) 大盘过滤：检查 `Index_Status`，确定 `Strategy Coefficient`（1.0 / 0.8 / 0.0）  
3) 指标计算：向量化计算 7 维指标及 `Score`  
4) 否决检查：  
   - 先查 Hard Veto → 若命中，`Score = 0`  
   - 再查 Soft Veto → 若命中，`Score = min(Score, 0.4)`  
5) 总分汇总：  
   - `Final_Score = Σ(Score_i * w_i) * Strategy_Coefficient`

### 5.2 交易指令映射

| Final Score | 决策建议 | 仓位管理 |
|---:|---|---|
| ≥ 0.80 | 强力买入 | 单票 20% - 30%（重仓） |
| 0.65 ~ 0.79 | 试探买入 | 单票 10% - 15%（底仓） |
| 0.40 ~ 0.64 | 观望/持有 | 不开新仓；已有持仓可持有 |
| < 0.40 | 减仓/卖出 | 逐步离场 |
| Hard Veto（0） | 清仓 | 市价止损（Market Order） |

---

## 6. 参数配置表（Config）

建议将这些参数提取到 Python 的 `config.py` 文件中，方便后期调优。

```python
CONFIG = {
    # 趋势
    'MA_WINDOW': 20,
    'SLOPE_THRESHOLD_BULL': 0.0010,  # 0.10%
    'DEV_MAX': 0.06,                 # 乖离率警戒线 6%

    # RSI
    'RSI_PERIODS': [6, 12, 24],
    'RSI_LOW': 35,
    'RSI_HIGH': 75,
    'RSI_DEAD_LINE': 40,             # 强弱分界线

    # VR (量比)
    'VR_LIMIT_LOW': 0.6,
    'VR_LIMIT_HIGH': 6.0,
    'VR_BUY_MIN': 1.5,
    'VR_BUY_MAX': 3.5,
    'VR_EARLY_MAX': 5.0,             # 早盘 10:00 前宽容度
    'VR_EARLY_SOFT_LOW': 0.4,
    'VR_EARLY_SOFT_HIGH': 8.0,
    'VR_SOFT_CONFIRM_N': 2,
    'EARLY_START': "09:30",
    'EARLY_END': "10:00",

    # BOLL
    'BOLL_PERIOD': 20,
    'BOLL_STD': 2,

    # 逻辑控制
    'LOOKBACK_DAYS': 5,              # 背离回溯天数
    'MIN_K_BARS': 200,               # 最小数据长度
    'RET_ABS_MAX': 0.25,
    'MAX_MISSING_RATIO': 0.02,
    'MAX_GAP_DAYS': 3
}
```
