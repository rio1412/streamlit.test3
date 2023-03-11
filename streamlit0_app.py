%%writefile app.py 

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


@st.cache(show_spinner=False)
def load_image(image_file):
    img = tf.io.decode_image(image_file.read())
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    return img[tf.newaxis, :]


@st.cache(show_spinner=False)
def apply_style(content_image, style_image):
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image


st.title('画風変換アプリ')

content_image_file = st.file_uploader('コンテンツ画像をアップロードしてください')
style_image_file = st.file_uploader('スタイル画像をアップロードしてください')

if content_image_file and style_image_file:
    content_image = load_image(content_image_file)
    style_image = load_image(style_image_file)
    stylized_image = apply_style(content_image, style_image)
    st.image(stylized_image, caption='結果')    
