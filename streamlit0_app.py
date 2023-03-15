import streamlit as st
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# テキストを分かち書きする関数
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    return [token.surface for token in tokens]

# ワードクラウドを作成する関数
def create_wordcloud(words, mask_image=None):
    if mask_image is not None:
        mask = np.array(mask_image)
    else:
        mask = None
    wordcloud = WordCloud(background_color="white",
                          width=800, height=600,
                          font_path="./font/mplus-2p-regular.ttf",
                          contour_width=2, contour_color='steelblue',
                          mask=mask
                          ).generate(' '.join(words))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot()

# Streamlitアプリの設定
st.set_page_config(page_title="分かち書き＆ワードクラウドアプリ", page_icon=":pencil:", layout="wide")

# テキスト入力
st.sidebar.header("テキスト入力")
text = st.sidebar.text_area("ここにテキストを入力してください。")

# マスク画像アップロード
st.sidebar.header("マスク画像")
mask_image = st.sidebar.file_uploader("ここにマスク画像をドロップしてください", type=["jpg", "jpeg", "png"])

# 分かち書きとワードクラウドの作成
if text:
    st.header("分かち書き結果")
    words = tokenize(text)
    st.write(words)

    st.header("ワードクラウド")
    if mask_image is not None:
        mask = Image.open(mask_image)
        create_wordcloud(words, mask_image=mask)
    else:
        create_wordcloud(words)
else:
    st.write("テキストを入力してください。")
