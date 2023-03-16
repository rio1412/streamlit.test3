import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# モデルをロードする
model = load_model('path/to/your/model.h5')

# 分類するための関数を定義する
def predict(image):
    # 画像をリサイズする
    image = cv2.resize(image, (224, 224))
    # 画像を正規化する
    image = image.astype('float32') / 255
    # モデルを使用して予測する
    predictions = model.predict(np.expand_dims(image, axis=0))[0]
    # 最も確率の高いクラスを取得する
    class_idx = np.argmax(predictions)
    # クラスのラベルを取得する
    labels = ['apple_pie', 'baby_back_ribs', 'baklava', ...] # データセットのラベルを定義する
    label = labels[class_idx]
    # クラスの確率を取得する
    probability = predictions[class_idx]
    return label, probability

# Streamlitアプリを作成する
def main():
    st.title('料理レシピアプリ')
    # 画像をアップロードするためのボタンを作成する
    uploaded_file = st.file_uploader("食材の画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 画像を読み込む
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # 画像を分類する
        label, probability = predict(image)
        # 分類された食材に基づいてレシピを表示する
        if label == 'apple_pie':
            st.write('りんごパイのレシピを表示する')
        elif label == 'baby_back_ribs':
            st.write('ベビーバックリブのレシピを表示する')
        elif label == 'baklava':
            st.write('バクラバのレシピを表示する')
        ...
        # 分類結果を表示する
        st.write('この画像は「' + label + '」と予測されました。')
        st.write('確率: {:.2f}%'.format(probability * 100))

if __name__ == '__main__':
    main()
