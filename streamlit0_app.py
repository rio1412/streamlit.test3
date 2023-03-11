import random
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# アプリのタイトルを表示
st.title("Financial Data Visualization App")

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

    # 経常利益と株価の散布図を描画する関数
    def plot_scatter(x, y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_title("Scatter plot")
        return fig

    # 経常利益と営業利益率の折れ線グラフを描画する関数
    def plot_line(x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_title("Line plot")
        return fig

    # 純資産と売上高の成長率の棒グラフを描画する関数
    def plot_bar(x, y):
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_title("Bar plot")
        return fig

    # 表示する図をランダムに選択する
    figures = []
    figure_functions = [plot_scatter, plot_line, plot_bar]
    for i in range(3):
        figure_func = random.choice(figure_functions)
        figure = figure_func(subset_data.iloc[:,i], subset_data.iloc[:,(i+1)%3])
        figures.append(figure)

    # 選択された図を表示する
    for figure in figures:
        canvas = FigureCanvas(figure)
        with st.spinner('Plotting...'):
            st.write(figure)
        st.write('\n')
