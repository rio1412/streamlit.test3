import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# タイトルの表示
st.title('ローン審査用財務会計分析アプリ')

# ファイルのアップロード
uploaded_file = st.file_uploader('ファイルをアップロードしてください', type=['csv'])

if uploaded_file is not None:
    # データの読み込み
    data = pd.read_csv(uploaded_file)
    
    # 可視化
    fig, ax = plt.subplots()
    ax.plot(data['年度'], data['売上高'], label='売上高')
    ax.plot(data['年度'], data['経常利益'], label='経常利益')
    ax.plot(data['年度'], data['自己資本比率'], label='自己資本比率')
    ax.legend()
    st.pyplot(fig)
    
    # 指標の計算
    sales_growth_rate = ((data['売上高'].iloc[-1] / data['売上高'].iloc[0]) ** (1/(len(data)-1))) - 1
    roa = data['経常利益'].sum() / data['総資産'].iloc[-1]
    equity_ratio = data['自己資本比率'].iloc[-1]
    
    # 結果の表示
    st.write('売上高成長率:', round(sales_growth_rate, 2))
    st.write('ROA:', round(roa, 2))
    st.write('自己資本比率:', round(equity_ratio, 2))
