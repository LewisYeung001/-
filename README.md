# SBCF
基于资金流向的支持向量机策略

简介：
资金的流入流出是影响股价短期波动的重要因素，因此，本文尝试对个股资金流向的建模，以提高股价预测的精度。而传统的时间序列分析方法中，又大多偏向于线性的分析方法，而忽略了股价波动的非线性因素。因此，本文利用模式识别中常用的一种感知器——支持向量机对资金流向进行建模，以预判下一交易日个股股价是上升还是下跌，并以此为依据进行股票交易。

主要流程：
首先考虑止损策略，当大盘(沪深300指数)的20日线死叉30日线时，清仓并保持空仓状态。而当大盘指数没有发出止损信号时，策略开始进行执行交易。
其次，在没有止损信号时，策略会先选择股票池。股票池是用来选择其中下一个交易日可能上升的股票的备选列表。策略会去除掉次新股、ST、退市整理板、创业板的股票。
再次，通过遍历的手段，对每一只股票的资金流向数据进行获取和数据清洗，以及对数据进行归一化处理。同时，为了便于训练，需要给股票每日的数据加标签。标签是为了模型学习历史数据用的。当第i个交易日证券j的涨幅超过0.5%时，则标记该证券的i-1交易日的标签为1，即为“上涨”；反之标记0，即为“下跌”。
再次，对整理好的数据进行SVM训练，本文对每只股票过去120日的资金流向数据进行训练。并得出对下一交易日的预测。如果预测为涨，即SVM预测模型返回值“1”，并且该证券的5日均线金叉10日均线，则将该股票放入“买入池”中。反之，返回为0时，则卖出该证券，如果没有持有该证券，则不进行操作。
最后，买进按照等股票数的方式（即每种股票的持股数相等）买进“买入池”中全部股票。

准备：
Python 3.x
Numpy >= 1.12.0
Scikit-learn >= 0.18.1
Pandas >= 0.12.0
JoinQuant或广发量化平台
您可以使用：
pip install xxx
即可安装

附注：广发量化平台API说明
https://quant.gf.com.cn/api

回测情况：
时间：2010/1/4-2018/7/30
总收益：523.04%
策略年化收益：24.57%
基准收益：-1.10%
Alpha：0.213
Beta：0.173
Sharpe：0.850
Sortino：1.109
Information Ratio：0.805
Algorithm Volatility：0.242
Benchmark Volatility：0.233
胜率：0.408
日胜率：0.487
盈亏比：2.534
盈利次数：102
亏损次数：148
最大回撤：43.025%
最大回撤出现时间：2015/06/02-2015/06/26

时间复杂度：
超过2100个交易日
需要时间：454.68s
