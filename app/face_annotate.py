from re import A
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from config import Emoji

BBox = tuple[int, int, int, int] # top, right, bottom, left

# -> : 返り値の型を明確にするための記述
def get_emoji_size(bbox: BBox) -> int:
    top, right, bottom, left = bbox
    size = max(bottom - top, right - left)
    return size


def get_emoji_position(bbox: BBox) -> tuple[int, int]:
    top, _, _, left = bbox
    return (left, top)


def draw_emoji(im: Image, emoji: Emoji, size: int, pos: tuple[int, int]) -> None:
    emoji_path = 'app/emoji/' + f"{emoji.name}.png"
    emoji_im = Image.open(emoji_path).resize((size, size)).convert("RGBA")
    im.paste(emoji_im, pos, emoji_im)


def face_detection(im: Image) -> list[BBox]:
    image = np.asarray(im)
    face_bboxes = face_recognition.face_locations(image)
    return face_bboxes


def get_image_face_hided_by_emoji(im: Image, emoji: Emoji) -> Image:
    face_bboxes = face_detection(im)
    for face_bbox in face_bboxes:
        size = get_emoji_size(face_bbox)
        pos = get_emoji_position(face_bbox)
        draw_emoji(im, emoji, size, pos)

    return im

def draw_face(img, locs) -> Image:
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, mode='RGBA')

    for top, right, bottom, left in locs:
        draw.rectangle((left, top, right, bottom), outline='lime', widht=2)

    return img