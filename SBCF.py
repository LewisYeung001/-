# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:11:58 2018

SBCF Model
Based on JoinQuant

@author: Lewis Yeung
"""

# 导入函数库
import jqdata
from sklearn import svm

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000985.XSHG')
    # 初始化买入股票池
    g.rise = {'a':0}
    # 打开性能分析，以便于分析算法复杂度
    enable_profile()
    
## 导入资金流数据，并进行模型训练
def training(s,context):
    pred = 0
    # 获取某只股票的资金流数据
    moneyflow = jqdata.get_money_flow(s,end_date = context.current_dt.date(),count=120)
    
    # 数据检查
    if len(moneyflow)<25:
        return 0
    
    # 为标准化做准备
    # 引入主力净额和主力净占比，标准化是防止出现拟合失败的问题
    amax = moneyflow['net_amount_main'].max()
    amin = moneyflow['net_amount_main'].min()
    amean = moneyflow['net_amount_main'].mean()
    astd = moneyflow['net_amount_main'].std()

    pmax = moneyflow['net_pct_main'].max()
    pmin = moneyflow['net_pct_main'].min()
    pmean = moneyflow['net_pct_main'].mean()    
    pstd = moneyflow['net_pct_main'].std()
    
    X = []
    Y = []
    
    i = 4
    
    # 利用Max-Min标准化后的的数据做训练
    while i < len(moneyflow) - 5:
        
        if moneyflow['change_pct'][i] > 0.5:
            Y.append(1)
        else:
            Y.append(0)

        x1 = (moneyflow['net_amount_main'][i-1] - amin) / (amax - amin)
        x2 = (moneyflow['net_amount_main'][i-2] - amin) / (amax - amin)
        x3 = moneyflow['change_pct'][i-1] / 10
        x4 = (moneyflow['net_pct_main'][i-1] - pmin) / (pmax - pmin)

        X.append([x1,x2,x3,x4])
        
        i = i + 1
    
    # 利用sklearn模块进行SVM模型训练
    clf = svm.SVC()
    clf.fit(X, Y)  
    
    #预测
    x1 = (moneyflow['net_amount_main'][len(moneyflow)-1] - amin) / (amax - amin)
    x2 = (moneyflow['net_amount_main'][len(moneyflow)-2] - amin) / (amax - amin)
    x3 = moneyflow['change_pct'][len(moneyflow)-1] / 10
    x4 = (moneyflow['net_pct_main'][len(moneyflow)-1] - pmin) / (pmax - pmin)
    X1 = [x1,x2,x3,x4]

    pred = clf.predict(X1)
    
    return pred

## 构建股票池

# 过滤ST、*ST和退市整理板股票
def filter_paused_and_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused 
    and not current_data[stock].is_st and 'ST' not in current_data[stock].
    name and '*' not in current_data[stock].name and '退' not in current_data[stock].name]
 
# 过滤掉创业板股票   
def filter_gem_stock(context, stock_list):
    return [stock for stock in stock_list  if stock[0:3] != '300']

# 过滤掉次新股
def filter_old_stock(context, stock_list):
    tmpList = []
    for stock in stock_list :
        days_public=(context.current_dt.date() - get_security_info(stock).start_date).days
        # 上市超过1年
        if days_public > 365:
            tmpList.append(stock)
    return tmpList

# 过滤涨跌停股票    
def filter_limit_stock(context, data, stock_list):
    tmpList = []
    curr_data = get_current_data()
    for stock in stock_list:
        # 未涨停，也未跌停
        if curr_data[stock].low_limit < data[stock].close < curr_data[stock].high_limit:
            tmpList.append(stock)
    return tmpList

# 筛选出股票池
def select_stocks(context):
    # 选取流通市值小于100亿的100只股票
    q = query(valuation.code, valuation.circulating_market_cap).order_by(
            valuation.circulating_market_cap.asc()).filter(
            valuation.circulating_market_cap <=100).limit(50)
    df = get_fundamentals(q)
    stock_list = list(df['code'])
    
    # 过滤掉停牌的和ST的
    stock_list = filter_paused_and_st_stock(stock_list)
    #过滤掉创业板
    stock_list = filter_gem_stock(context, stock_list)
    # 过滤掉上市小于1年的
    stock_list = filter_old_stock(context, stock_list)
    # 选取前N只股票放入“目标池”
    stock_list = stock_list[:30]  
    return stock_list
    
## 止损函数，当沪深300指数的20日线下穿30日线时，卖出所有股票
def stop(context):
    Price = attribute_history('000300.XSHG',60,'1d', ['close'])
    ma30 = Price['close'][-30:-1].mean()
    ma20 = Price['close'][-20:-1].mean()
    if ma20<ma30:
        return False;
    else:
        return True
    
## 交易
def before_market_open(context):
    
    # 清空字典，初始化买入股票池
    g.rise.clear()
    # 股票池
    g.df = select_stocks(context)
    # 如果止损信号发出
    if stop(context) == False:
        #卖掉所有股票
        for s in context.portfolio.positions:
            order_target(s,0)
    # 如果止损信号未发出
    elif stop(context) == True:
        for s in g.df:
            predict = training(s,context)
            if predict == 1:
                # 预测为涨，则计算某只股票的5日、10日移动平均，金叉则放入买入池
                Price = attribute_history(s,20,'1d', ['close'],skip_paused=False)
                ma5 = Price['close'][-5:-1].mean()
                ma10 = Price['close'][-10:-1].mean()
                if ma5 > ma10:
                    price = Price['close'][-1]
                    g.rise[s] = price
            # 如果预测为跌，卖出所持的该股票
            elif predict == 0 and context.portfolio.positions[s].closeable_amount > 0:
                # 全部卖出
                order_target(s, 0)
                # 记录这次卖出
                log.info("Selling %s" % (s))
    
        # 用所有可用 cash 买进买入池的股票
        cash = context.portfolio.available_cash
        total_price = sum(g.rise.values())
        volume = cash/total_price

        # 下单
        for s in g.rise.keys(): #s是股票list的遍历
            order_target(s, volume)
            # 记录这次买入
            log.info("Buying %s" % (s))
    
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))
    
