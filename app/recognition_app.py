from enum import Enum
from PIL import Image
import numpy as np
import streamlit as st
from config import Emoji
from face_annotate import get_image_face_hided_by_emoji, face_detection, draw_face

st.title('顔を絵文字で隠せるアプリ')

upload_file = st.file_uploader('画像ファイルの選択')
# サンプル画像
image_paths = {
    'app/img/asuka.jpeg': "齋藤飛鳥",
    'app/img/elon.jpeg': "イーロン・マスク",
}

example_image = st.selectbox(
    "サンプル画像",
    list(image_paths.keys()),
    format_func=lambda x: image_paths[x]
)

# 顔文字選択
emoji = st.selectbox(
    "絵文字",
    list(Emoji),
    format_func=lambda x: f"{x.value} "
)

page_left, page_right = st.columns([1,1])
page_left.subheader('入力画像')
page_right.subheader('実行結果')

image = Image.open(example_image)
if upload_file is not None:
    image = Image.open(upload_file)

page_left.image(image)

if page_left.button('実行'):
    with st.spinner('実行中...'):
        output = get_image_face_hided_by_emoji(image, emoji)
        page_right.image(output)

