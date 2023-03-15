import streamlit as st
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_wordcloud(text):
    # 形態素解析
    t = Tokenizer()
    words = t.tokenize(text)
    # 単語の出現回数をカウント
    word_count = {}
    for word in words:
        if word.surface not in word_count:
            word_count[word.surface] = 1
        else:
            word_count[word.surface] += 1
    # ワードクラウドの作成
    wordcloud = WordCloud(
        background_color="white",
        font_path="0D10DE1D-5D14-4E5C-B492-0164B69F4AB4.png",
        width=800,
        height=800,
        prefer_horizontal=1.0,
        min_word_length=2,
        max_font_size=120,
        colormap='coolwarm',
        mask=np.array(Image.open('./img/mask.png')),
        contour_width=3,
        contour_color='black'
    ).generate_from_frequencies(word_count)

    # ワードクラウドを表示
    plt.figure(figsize=[8, 8])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

def main():
    st.title("JanomeとWordCloudを組み合わせたアプリ")
    st.write("以下にテキストを入力してください。")
    text = st.text_area("テキスト入力", "", height=200)
    if st.button("ワードクラウドを生成"):
        create_wordcloud(text)

if __name__ == "__main__":
    main()
