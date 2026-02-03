CONFIG = {
    # 基础路径/连接
    'QMT_USERDATA_PATH': r'E:\\Software\\stock\\gjzqqmt\\国金证券QMT交易端\\userdata_mini',
    'QMT_ACCOUNT_ID': '',
    'QMT_ACCOUNT_TYPE': 'STOCK',

    # 扫描范围
    'SCAN_SECTOR': '沪深A股',
    'TEST_SAMPLE': 0,

    # 行情取数
    'DIVIDEND_TYPE': 'front',
    'DAY_COUNT': 60,
    'MINUTE_COUNT': 1200,
    'RSI_COUNT': 200,

    # 连接与缓存
    'PORT': 58610,
    'ENABLE_DOWNLOAD': False,
    'CACHE_DIR': './temp_cache',

    # 输出
    'SAVE_DIR': './output',

    # 趋势
    'MA_WINDOW': 20,
    'SLOPE_THRESHOLD_BULL': 0.10,    # Slope20 > 0.10 (%) 记为趋势向上
    'SLOPE_FLAT_MIN': 0.0,           # 走平下限
    'SLOPE_FLAT_MAX': 0.10,          # 走平上限
    'SLOPE_VETO': -0.5,              # 硬否决阈值（%）
    'SLOPE_SELL_FLAT_MIN': -0.10,    # 卖出“走平/减弱”范围
    'SLOPE_SELL_FLAT_MAX': 0.10,
    'DEV_MAX': 0.06,                 # 乖离率警戒线 6%
    'DEV_MAX_EXT': 0.10,             # 强趋势但乖离更大
    'MA20_NEAR_PCT': 0.02,           # 接近MA20的范围（2%）

    # RSI
    'RSI_PERIODS': [6, 12, 24],
    'RSI_LOW': 35,
    'RSI_HIGH': 75,
    'RSI_DEAD_LINE': 40,             # 强弱分界线
    'RSI_BUY_MIN': 55,
    'RSI_BUY_MAX': 80,
    'RSI_SELL_HIGH': 85,
    'RSI_SELL_LOW': 45,
    'RSI_MID': 50,
    'RSI_SELL_LOW_BUFFER': 5,

    # VR (量比)
    'VR_LIMIT_LOW': 0.6,
    'VR_LIMIT_HIGH': 6.0,
    'VR_BUY_MIN': 1.5,
    'VR_BUY_MAX': 3.5,
    'VR_EARLY_MAX': 5.0,             # 早盘 10:00 前宽容度
    'VR_BUY_PART_MIN': 1.0,
    'VR_BUY_PART_MAX': 1.5,
    'VR_VETO_LOW': 0.5,
    'VR_VETO_HIGH': 5.0,
    'VR_SELL_FULL': 0.8,
    'VR_SELL_PART': 4.0,
    'VR_SELL_VETO_DOWN_PCT': -9.5,
    'VR_SELL_VETO_VOL': 1.5,
    'VR_EARLY_SOFT_LOW': 0.4,
    'VR_EARLY_SOFT_HIGH': 8.0,
    'VR_SOFT_CONFIRM_N': 2,
    'EARLY_START': "09:30",
    'EARLY_END': "10:00",

    # BOLL
    'BOLL_PERIOD': 20,
    'BOLL_STD': 2,
    'BOLL_TOUCH_SCORE': 0.4,

    # KDJ
    'KDJ_K_MAX_BUY': 80,
    'KDJ_K_NEAR_MIN': 75,
    'KDJ_K_NEAR_MAX': 80,
    'KDJ_K_VETO': 90,
    'KDJ_J_VETO': 100,
    'KDJ_J_FLAT_EPS': 0.1,

    # 评分分值（买入/卖出）
    'SCORE_FULL': 1.0,
    'SCORE_BUY_TREND_FLAT': 0.6,
    'SCORE_BUY_MA_PART': 0.4,
    'SCORE_BUY_MACD_PART': 0.6,
    'SCORE_BUY_KDJ_PART': 0.3,
    'SCORE_BUY_RSI_PART1': 0.8,
    'SCORE_BUY_RSI_PART2': 0.5,
    'SCORE_BUY_BOLL_PART': 0.6,
    'SCORE_BUY_VOL_PART': 0.6,

    'SCORE_SELL_TREND_PART': 0.6,
    'SCORE_SELL_MA_PART': 0.4,
    'SCORE_SELL_MACD_PART': 0.5,
    'SCORE_SELL_KDJ_PART': 0.7,
    'SCORE_SELL_RSI_PART': 0.6,
    'SCORE_SELL_VOL_PART': 0.8,

    # 权重（总分）
    'WEIGHT_TREND': 0.30,
    'WEIGHT_MACD': 0.20,
    'WEIGHT_RSI': 0.20,
    'WEIGHT_KDJ': 0.15,
    'WEIGHT_BOLL': 0.10,
    'WEIGHT_VR': 0.05,

    # 总分阈值
    'THRESH_STRONG_BUY': 0.80,
    'THRESH_TRY_BUY_MIN': 0.65,
    'THRESH_TRY_BUY_MAX': 0.79,
    'THRESH_HOLD_MIN': 0.40,
    'THRESH_HOLD_MAX': 0.64,

    # 逻辑控制
    'LOOKBACK_DAYS': 5,              # 背离回溯天数
    'MIN_K_BARS': 200,               # 最小数据长度
    'RET_ABS_MAX': 0.25,
    'MAX_MISSING_RATIO': 0.02,
    'MAX_GAP_DAYS': 3
}
