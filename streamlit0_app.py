import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import numpy as np
from PIL import Image

# Arbitrary Image Stylizationモデルの読み込み
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# 画像の前処理関数
def preprocess_img(img):
    img = img.resize((256, 256))  # 画像サイズをリサイズ
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = img[tf.newaxis, ...]
    return img

# 画像の後処理関数
def deprocess_img(img):
    img = np.array(img)
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
    img = img.astype('uint8')
    return Image.fromarray(img)

# 画像の読み込み関数
def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = preprocess_img(img)
    return img

# スタイル画像とコンテンツ画像の取得と前処理
def get_style_content_images(style_image_file, content_image_file):
    style_image = load_image(style_image_file)
    content_image = load_image(content_image_file)
    return style_image, content_image

# 画風変換を実行する関数
@st.cache
def stylize(style_image, content_image):
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    return deprocess_img(stylized_image)

# メイン関数
def main():
    # タイトル
    st.title("Arbitrary Image Stylization")

    # スタイル画像とコンテンツ画像のアップロード
    st.header("Upload Style and Content Images")
    style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])

    if style_image_file and content_image_file:
        # スタイル画像とコンテンツ画像の取得と前処理
        style_image, content_image = get_style_content_images(style_image_file, content_image_file)

        # スタイル画像とコンテンツ画像の表示
        st.header("Style Image")
        st.image(style_image.numpy(), use_column_width=True)

        st.header("Content Image")
        st.image(content_image.numpy(), use_column_width=True)

        # 画風変換を実行
        stylized_image = stylize(style_image, content_image)

        # 結果の表示
        st.header("Stylized Image")
        st.image(stylized_image, use_column_width=True)

if __name__ == '__main__':
    main()
