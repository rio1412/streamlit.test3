import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Declaring Constants
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def load_model():
    model = hub.load(SAVED_MODEL_PATH)
    return model

model = load_model()

# 新たに必要なimport文を追加
from itertools import cycle

def plot_images(images, titles):
  """
    Plots images from a list of image tensors.
    Args:
      images: List of 3D image tensors. Each tensor has shape [height, width, channels].
      titles: List of titles to display in the plot.
  """
  # 各画像をnumpy arrayに変換し、クリッピングする
  images = [tf.clip_by_value(np.asarray(image), 0, 255) for image in images]
  # PIL Imageに変換する
  images = [Image.fromarray(tf.cast(image, tf.uint8).numpy()) for image in images]
  # 画像とタイトルをサイクルしてループする
  for image, title in zip(cycle(images), cycle(titles)):
    st.image(image, caption=title, use_column_width=True)

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image, caption="Original Image", use_column_width=True)
    hr_image = preprocess_image(input_image)

    if st.button('Enhance image'):
        if hr_image is not None:
            # Loading the model
            start = time.time()
            fake_images = []
            # 色違い画像を生成するために、hueの値を変えながら複数回スーパーリゾリューションを実行する
            for i in range(3):
                hue_i = hue + i * 0.2 # hueを0.2ずつ増加させる
                fake_image = model(tf.image.adjust_hue(hr_image, hue_i))
                fake_image = tf.squeeze(fake_image)
                fake_images.append(fake_image)
            st.write("Time Taken : ", time.time() - start)

            # Displaying the Super Resolution Images
            st.write("")
            st.write("## Super Resolution")
            st.write("")
            plot_images(fake_images, titles=["Super Resolution #1", "Super Resolution #2", "Super Resolution #3"])

            # Saving the Super Resolution Images
            for i, fake_image in enumerate(fake_images):
                save_image(tf.squeeze(fake_image), filename=f"Super Resolution #{i+1} Adjusted")

  hr_image = np.array(image)
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    # Convert the image to an RGB mode if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  st.image(image, caption=title, use_column_width=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Super Resolution")
st.sidebar.title("Settings")
contrast = st.sidebar.slider('Contrast', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
st.sidebar.write('Contrast:', contrast)

brightness = st.sidebar.slider('Brightness', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
st.sidebar.write('Brightness:', brightness)

gamma = st.sidebar.slider('Gamma', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
st.sidebar.write('Gamma:', gamma)

hue = st.sidebar.slider('Hue', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
st.sidebar.write('Hue:', hue)

image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image, caption="Original Image", use_column_width=True)
    hr_image = preprocess_image(input_image)

    if st.button('Enhance image'):
        if hr_image is not None:
            # Loading the model
            start = time.time()
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)
            st.write("Time Taken : ", time.time() - start)

            # Displaying the Super Resolution Image
            st.write("")
            st.write("## Super Resolution")
            st.write("")

            # Applying Contrast, Brightness and Gamma Correction
            fake_image = tf.image.adjust_contrast(fake_image, contrast)
            fake_image = tf.image.adjust_brightness(fake_image, brightness)
            fake_image = tf.image.adjust_gamma(fake_image, gamma)
            fake_image = tf.image.adjust_hue(fake_image, hue)




            # Displaying the Super Resolution Image with adjusted color and contrast
            plot_image(tf.squeeze(fake_image), title="Super Resolution with Adjusted Color and Contrast")

            # Saving the Super Resolution Image with adjusted color and contrast
            save_image(tf.squeeze(fake_image), filename="Super Resolution Adjusted")

else:
    st.write("Upload an image to get started.")
