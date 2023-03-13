import streamlit as st
from pydub import AudioSegment
from scipy import fftpack
import numpy as np
import cv2
from IPython.display import display, HTML
import time

st.title("音楽に合わせて明滅するアプリ")

uploaded_file = st.file_uploader("音楽ファイルをアップロードしてください", type=['mp3'])

if uploaded_file is not None:
    # 音楽ファイルを読み込む
    song = AudioSegment.from_file(uploaded_file)

    # プログレスバーを表示する
    progress_bar = st.progress(0)

    # 明滅のパターンを定義する
    pattern = [np.sin(np.linspace(0, 2*np.pi, 100)) * 127 + 128,
               np.sin(np.linspace(0, 2*np.pi, 100)) * -127 + 128,
               np.zeros(100)]

    # 明滅のインデックスを初期化する
    pattern_index = 0

    # 音楽を再生する
    song.export("tmp.wav", format="wav")
    display(HTML("""
        <audio controls>
          <source src="tmp.wav" type="audio/wav">
        </audio>
    """))
    time.sleep(1)  # 音楽ファイルが読み込まれるまで待機する

    # 明滅を表示するウィンドウを作成する
    cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Color', 400, 400)

    # フレームをループする
    samples = np.array(song.get_array_of_samples())
    for i in range(len(samples)):
        # 周波数成分の強度から色を決定する
        color = np.array([np.clip(spectrum[i], 0, 255), 255-np.clip(spectrum[i], 0, 255), 0])

        # 明滅のパターンに従って色を変更する
        color = np.array([color[j] + pattern[pattern_index][j] for j in range(3)])
        color = np.clip(color, 0, 255).astype(np.uint8)

        # 明滅のインデックスを更新する
        pattern_index += 1
        if pattern_index >= len(pattern):
            pattern_index = 0

        # ウィンドウに色を描画する
        cv2.imshow('Color', np.tile(color, (400, 400, 1)))
        cv2.waitKey(1)

        # プログレスバーを更新する
        progress_bar.progress(i / len(samples))

    # ウィンドウを閉じる
    cv2.destroyAllWindows()
