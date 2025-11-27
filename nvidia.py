import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. データ取得 & 前処理
# ---------------------------------------------------------
def load_returns(start, end):
    """
    YahooFinance から調整後終値を取得し、対数リターンを返す。
    """
    tickers = ['NVDA', 'SPY', 'SMCI', 'AMD', 'MSFT', 'XLV', 'XLU']

    print(f"株価データ取得: {start} → {end}")
    prices = yf.download(tickers, start=start, end=end)['Adj Close']

    # 欠損があれば前日の値で補完
    prices = prices.fillna(method='ffill').dropna()

    # 対数収益率
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ---------------------------------------------------------
# 2. Nvidia 固有ショックと各銘柄の反応 β 推定
# ---------------------------------------------------------
def run_two_stage_regression(returns):
    """
    まず NVDA の市場影響（SPY）を取り除き残差を"NVDA の固有ショック"とみなし、
    次に他銘柄がそのショックにどれだけ反応するかを推計する。
    """
    # --- Step1: NVDA の市場ベータ調整 ---
    X = sm.add_constant(returns['SPY'])
    y = returns['NVDA']

    step1 = sm.OLS(y, X).fit()

    nvda_resid = step1.resid
    nvda_resid.name = "NVDA_Shock"

    print("\n[Step1] NVDA の市場ベータを推定")
    print(f"  NVDA β to SPY: {step1.params['SPY']:.4f}")

    # --- Step2: 各銘柄の NVDA ショック感応度 ---
    factors = pd.concat([returns['SPY'], nvda_resid], axis=1)
    X2 = sm.add_constant(factors)

    results = []
    targets = [c for c in returns.columns if c not in ['NVDA', 'SPY']]

    for tkr in targets:
        model = sm.OLS(returns[tkr], X2).fit()
        results.append({
            "Ticker": tkr,
            "Market_Beta": model.params['SPY'],
            "NVDA_Sensitivity": model.params['NVDA_Shock'],
            "R2": model.rsquared
        })

    df = pd.DataFrame(results).set_index("Ticker")
    return df, nvda_resid


# ---------------------------------------------------------
# 3. Nvidia 下落時の市場インパクトの簡易試算
# ---------------------------------------------------------
def simulate_market_effect(result_df, nvda_move=-0.10, nvda_weight=0.07):
    """
    Nvidia が一定割合下落したときの S&P500 への影響を簡易的に推計する。
    """
    print(f"\n[Simulation] Nvidia が {nvda_move*100:.1f}% 下落した場合の想定")

    # SP500 へ直接入る影響
    direct = nvda_weight * nvda_move

    # 他銘柄が NVDA ショックに反応する分
    avg_beta = result_df['NVDA_Sensitivity'].mean()
    spillover = (1 - nvda_weight) * avg_beta * nvda_move * 1.5  # 仮に流動性係数1.5

    total = direct + spillover

    print(f"  直接影響 : {direct*100:.2f}%")
    print(f"  波及影響 : {spillover*100:.2f}%")
    print(f"  合計     : {total*100:.2f}%")
    if direct != 0:
        print(f"  マルチプライヤー: {total/direct:.2f} 倍")


# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
if __name__ == "__main__":

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    # データ読み込み
    ret = load_returns(start, end)

    # 回帰分析
    sensitivity, nvda_shock = run_two_stage_regression(ret)

    print("\n[Step2] 各銘柄の NVDA ショック感応度")
    print("-" * 50)
    print(sensitivity[['NVDA_Sensitivity', 'Market_Beta']].sort_values('NVDA_Sensitivity', ascending=False))
    print("-" * 50)

    # シミュレーション
    simulate_market_effect(sensitivity)

    # --- グラフ ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=sensitivity.index,
        y="NVDA_Sensitivity",
        data=sensitivity,
        palette="viridis"
    )
    plt.axhline(0, color='black')
    plt.title("Sensitivity to Nvidia Idiosyncratic Shock")
    plt.ylabel("Sensitivity")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
