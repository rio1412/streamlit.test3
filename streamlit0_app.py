import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt

# モデルの読み込み
hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

# 画像の前処理
def preprocess_image(img_path):
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    # Convert to grayscale
    img_gray = tf.image.rgb_to_grayscale(img)
    img_gray = tf.tile(img_gray, [1, 1, 3])
    
    # Resize image
    img_lr = tf.image.resize(img_gray, (int(img.shape[0]/4), int(img.shape[1]/4)), method='bicubic')
    
    # Resize image for bicubic downsample
    img_bicubic = tf.image.resize(img_lr, (img_lr.shape[0]*4, img_lr.shape[1]*4), method='bicubic')
    
    # Normalize image
    img_lr = img_lr / 255.0
    img_bicubic = img_bicubic / 255.0
    
    # Extract patches
    patches_lr = tf.image.extract_patches(
        images=tf.expand_dims(img_lr, axis=0),
        sizes=[1, 48, 48, 1],
        strides=[1, 48, 48, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Reshape patches
    patches_lr = tf.reshape(patches_lr, [-1, 48, 48, 3])
    
    # Convert patches to tensor
    patches_lr = tf.transpose(patches_lr, [0, 3, 1, 2])
    
    # Extract patches for bicubic downsample
    patches_bicubic = tf.image.extract_patches(
        images=tf.expand_dims(img_bicubic, axis=0),
        sizes=[1, 192, 192, 1],
        strides=[1, 192, 192, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Reshape patches for bicubic downsample
    patches_bicubic = tf.reshape(patches_bicubic, [-1, 192, 192, 3])
    
    # Convert patches for bicubic downsample to tensor
    patches_bicubic = tf.transpose(patches_bicubic, [0, 3, 1, 2])
    
    return patches_lr, patches_bicubic

# 画像の高画質化
def super_resolve(img_path, model):
    # 画像の前処理
    patches_lr, patches_bicubic = preprocess_image(img_path)
    
    # 高画質化の実行
    patches_sr = model(patches_lr)
    
    # 高画質化されたパッチを画像に結合
    # 高画質化されたパッチを画像に結合
    sr_img = tf.transpose(patches_sr, [0, 2, 3, 1])
    sr_img = tf.reshape(sr_img, [-1, tf.shape(patches_lr)[1]*4, tf.shape(patches_lr)[2]*4, 3])

    bicubic_img = tf.transpose(patches_bicubic, [0, 2, 3, 1])
    bicubic_img = tf.reshape(bicubic_img, [-1, tf.shape(patches_lr)[1]*4, tf.shape(patches_lr)[2]*4, 3])

    # 元の画像を読み込み
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像の表示
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(img)
    axs[0].set_title('Original')
    axs[1].imshow(bicubic_img.numpy()[0])
    axs[1].set_title('Bicubic')
    axs[2].imshow(sr_img.numpy()[0])
    axs[2].set_title('ESRGAN')

    plt.show()
