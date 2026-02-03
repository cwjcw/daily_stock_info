# -*- coding: utf-8 -*-
from xtquant import xtdata

# 1. 订阅行情（只有订阅了，QMT才会源源不断推数据给你）
stock_code = '601229.SH'  # 以你之前关注的上海银行为例
xtdata.subscribe_quote(stock_code, period='1d', count=10)

# 2. 获取当前快照
res = xtdata.get_full_tick([stock_code])

if res:
    print(f"✅ 连接成功！{stock_code} 当前最新价: {res[stock_code]['lastPrice']}")
else:
    print("❌ 库没问题，但没拿到数据。请检查：QMT客户端是否已登录？行情是否已连接？")