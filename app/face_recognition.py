from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import streamlit as st
import cv2


st.write('testaa')
# 顔検出ライブラリ
# image_size=顔を検出して切り取るサイズ, margin=顔まわりの余白
mtcnn = MTCNN(image_size=160, margin=10).eval()
# 顔を切り取るためのモデル
resnet = InceptionResnetV1(pretrained='vggface2')

# 画像をアップロードするための画面
upload_file = st.file_uploader('画像ファイルの選択')
if upload_file is not None:
    image = Image.open(upload_file)
    # 顔データを160*160に切り抜き
    img_cropped = mtcnn(image)
    # img_ar = np.array(image)
    st.image(
        image, caption='アップロード イメージ',
        use_column_width=True
    )
    boxes, _ = mtcnn.detect(image)

# streamlitで画面作成
# アップロード機能で画像input
# 顔認識して矩形(バウンディングボックス)表示
# 矩形表示部分(顔の位置)を絵文字で隠す