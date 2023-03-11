import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import requests
from io import BytesIO

# 画風変換の関数
@st.cache(show_spinner=False)
def stylize(style_image, content_image):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    return np.array(stylized_image[0])

# スタイル画像とコンテンツ画像の取得と前処理
def get_style_content_images(style_image_file, content_image_file):
    style_image = PIL.Image.open(style_image_file).convert('RGB').resize((256, 256))
    content_image = PIL.Image.open(content_image_file).convert('RGB').resize((256, 256))
    style_image = np.array(style_image) / 255.0
    content_image = np.array(content_image) / 255.0
    return style_image, content_image

# Streamlit アプリケーション
def app():
    # ウィジェット
    st.title('画風変換アプリ')
    style_image_file_widget = st.sidebar.file_uploader('スタイル画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
    content_image_file_widget = st.sidebar.file_uploader('コンテンツ画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
    style_image_widget = st.empty()
    content_image_widget = st.empty()

    # 画風変換を実行して結果を表示する関数
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            style_image_file = style_image_file_widget.value
            content_image_file = content_image_file_widget.value
            if style_image_file is not None and content_image_file is not None:
                # スタイル画像とコンテンツ画像の取得と前処理
                style_image, content_image = get_style_content_images(style_image_file, content_image_file)

                # スタイル画像とコンテンツ画像の表示
                style_image_widget.image(style_image, caption='スタイル画像', use_column_width=True)
                content_image_widget.image(content_image, caption='コンテンツ画像', use_column_width=True)

                # 画風変換を実行
                stylized_image = stylize(style_image, content_image)

                # 結果の表示
                st.image(stylized_image, caption='変換結果', use_column_width=True)

    style_image_file_widget.observe(on_change)
    content_image_file_widget.observe(on_change)

if __name__ == '__main__':
    app()
