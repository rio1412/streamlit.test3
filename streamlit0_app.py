import pandas as pd
import streamlit as st

# ファイルをアップロードするためのウィジェットを表示する
uploaded_file = st.file_uploader("Choose a file", type="csv")

# ファイルがアップロードされた場合、データを読み込む
if uploaded_file is not None:
    # ファイルを読み込む
    financial_data = pd.read_csv(uploaded_file, encoding="SHIFT-JIS")
    # データを表示する
    st.write(financial_data)

# 経常利益、営業利益率、株価のみのデータを抽出する
subset_data = financial_data[["経常利益","営業利益率","株価"]]

# グラフを描画する関数を定義
def plot_graph(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_title("Scatter plot")
    st.pyplot(fig)

# アプリのタイトルを表示
st.title("Financial Data Visualization App")

# データフレームを表示
st.write(financial_data)

# 経常利益、営業利益率、株価のみのデータを表示
st.write(subset_data)

# 売上高の成長率を計算して表示
net_assets = financial_data["純資産"]
net_assets_shift = net_assets.shift(-1)
growth_rate = (net_assets - net_assets_shift) / net_assets_shift * 100
gr_dropna = growth_rate.dropna()
st.write("売上高の成長率: ", gr_dropna)

# 加重平均を計算して表示
sum_wa = 0
count = 0

for year in reversed(gr_dropna):
    count += 1
    weight = year * count
    sum_wa += weight

result = sum_wa / ((1/2)*count*(count+1))
st.write("加重平均: ", result)

# 経常利益と株価の相関係数を計算して表示
correlation = financial_data[["経常利益","株価"]]
corr_val = correlation.corr().iloc[0, 1]
st.write("経常利益と株価の相関係数: ", corr_val)

# 経常利益と株価の散布図を描画
plot_graph(subset_data["経常利益"], subset_data["株価"])
