import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# アプリのタイトルを表示
st.title("Financial Predictions App")

# ファイルをアップロードするためのウィジェットを表示する
uploaded_file = st.file_uploader("Choose a file", type="csv")

# ファイルがアップロードされた場合、データを読み込む
if uploaded_file is not None:
    # ファイルを読み込む
    financial_data = pd.read_csv(uploaded_file, encoding="SHIFT-JIS")
    # データを表示する
    st.write(financial_data)

    # 収益のみのデータを抽出する
    revenue_data = financial_data[["株価"]]

    # 過去の収益データを散布図として描画
    def plot_scatter(x, y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_title("Revenue scatter plot")
        st.pyplot(fig)

    plot_scatter(revenue_data.index, revenue_data["株価"])

    # 線形回帰モデルを作成し、将来の収益を予測する
    def predict_revenue(x):
        model = LinearRegression()
        model.fit(revenue_data.index.values.reshape(-1, 1), revenue_data["株価"])
        y_pred = model.predict([[x]])
        return y_pred[0]

    # 予測する年度を入力する
    prediction_year = st.number_input("Input the year to predict revenue for", value=2022, step=1)

    # 将来の収益を予測して表示する
    predicted_revenue = predict_revenue(prediction_year)
    st.write("Predicted revenue for", prediction_year, ": ", predicted_revenue)
