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

    # 列を選択するウィジェットを表示する
    selected_columns = st.multiselect("Select columns to plot", options=list(financial_data.columns))
    
    if len(selected_columns) > 0:
        # 選択された列のデータを抽出する
        selected_data = financial_data[selected_columns]

        # 散布図を描画する
        def plot_scatter(x, y):
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.set_xlabel(x.name)
            ax.set_ylabel(y.name)
            ax.set_title("Scatter plot")
            st.pyplot(fig)

        for column in selected_columns:
            plot_scatter(financial_data.index, financial_data[column])

        # 線形回帰モデルを作成し、将来の値を予測する
        def predict_value(x, y, x_pred):
            model = LinearRegression()
            model.fit(x.values.reshape(-1, 1), y)
            y_pred = model.predict([[x_pred]])
            return y_pred[0]

        # 予測する年度を入力する
        prediction_year = st.number_input("Input the year to predict for", value=2022, step=1)

        for column in selected_columns:
            # 将来の値を予測して表示する
            predicted_value = predict_value(financial_data.index, financial_data[column], prediction_year)
            st.write("Predicted", column, "for", prediction_year, ": ", predicted_value)
