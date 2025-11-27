Nvidia市場波及モデル (Nvidia Market Impact Model)

このリポジトリは、レポート「課題6. 2026年に向けた米国市場におけるAIサイクルと政策転換の衝突」の付属プログラムです。
Nvidia (NVDA) の株価変動が、S&P 500および主要セクターに与える影響を定量的に分析するために作成されました。

分析の概要

本モデルは以下の手順で市場への波及効果を推計しています。

データ収集: Yahoo Finance APIを使用し、NVDA, SPY, および関連銘柄の過去2年間のデータを取得。

直交化 (Orthogonalization):

市場全体 (SPY) の影響を除去し、「純粋なNvidia固有のショック (Pure Nvidia Shock)」を抽出。

感応度分析 (Sensitivity Analysis):

各銘柄が「純粋なNvidiaショック」に対してどれだけ反応するか（感応度 $\beta$）を算出。

インパクト試算:

Nvidiaが $x\%$ 下落した際に、市場全体がどれだけ押し下げられるか（直接的寄与 + 連鎖的波及）をシミュレーション。

必要なライブラリ

このコードを実行するには、以下のPythonライブラリが必要です。

pip install pandas numpy yfinance statsmodels matplotlib seaborn


使用方法

ターミナルで以下を実行してください。

python nvidia_market_impact_model.py


実行すると、コンソールに分析結果（感応度やマルチプライヤー）が表示され、グラフが描画されます。

作成者

氏名: 吉田 友明

日付: 2025年11月27日

目的: 2026年市場展望レポートの定量的根拠として
