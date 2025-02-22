import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def pearsonr_demo():
    """
    相关系数计算
    :return: None
    """
    data = pd.read_csv("factor_returns.csv")
    plt.figure(figsize=(20, 8), dpi=100)
    plt.scatter(data['revenue'], data['total_expense'])
    plt.show()

    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
              'earnings_per_share', 'revenue', 'total_expense']

    for i in range(len(factor)):
        for j in range(i, len(factor) - 1):
            print(
                "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(data[factor[i]], data[factor[j + 1]])[0]))

    return None



