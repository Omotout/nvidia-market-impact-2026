import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 期間（とりあえず2年分）
end = datetime.now()
start = end - timedelta(days=365 * 2)

# 分析する銘柄
market = '^GSPC'   # S&P500
target = 'NVDA'    # 中心にみる銘柄
others = ['AMD', 'AVGO', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SMCI', '^SOX']

def load_returns(start, end):
    # まとめて取れるやつは全部取る
    tickers = [market, target] + others
    data = yf.download(tickers, start=start, end=end)
    
    # yfinance新バージョンではauto_adjust=Trueがデフォルト
    # Closeが調整後終値になっている
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = data[['Close']]
    
    data = data.dropna()

    # 対数リターン
    ret = np.log(data / data.shift(1))
    return ret.dropna()

def orthogonalize_nvda(ret):
    # Step1: NVDA の市場要因を落とす
    y = ret[target]
    X = sm.add_constant(ret[market])
    model = sm.OLS(y, X).fit()
    nvda_res = model.resid
    nvda_res.name = "nvda_res"
    return nvda_res, model

def sensitivity_analysis(ret, nvda_res):
    results = []

    # 市場とNVDA残差を並べる
    X = pd.concat([ret[market], nvda_res], axis=1)
    X = sm.add_constant(X)

    for t in others:
        if t not in ret.columns:
            continue

        y = ret[t]
        m = sm.OLS(y, X).fit()

        results.append({
            "Ticker": t,
            "Market_Beta": m.params[market],
            "NVDA_Sensitivity": m.params["nvda_res"],
            "P_Value": m.pvalues["nvda_res"],
            "R2": m.rsquared
        })

    return pd.DataFrame(results)

def plot_sensitivity(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Ticker", y="NVDA_Sensitivity", data=df, hue="Ticker", palette="viridis", legend=False)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Sensitivity to NVDA Idiosyncratic Shock")
    plt.tight_layout()
    plt.show()

def main():
    ret = load_returns(start, end)
    nvda_res, step1 = orthogonalize_nvda(ret)

    print("NVDAの市場ベータ:", round(step1.params[market], 4))

    df = sensitivity_analysis(ret, nvda_res)
    print(df[["Ticker", "NVDA_Sensitivity", "P_Value"]])

    # NVDAが10%動いた場合のざっくり計算
    shock = 0.10
    df["Impact_10pct"] = df["NVDA_Sensitivity"] * shock * 100
    print("\nNVDAが10%動いた時の推定インパクト（%）:")
    print(df[["Ticker", "Impact_10pct"]].round(2))

    plot_sensitivity(df)

if __name__ == "__main__":
    main()

