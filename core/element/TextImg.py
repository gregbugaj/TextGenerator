import cv2
from typing import List, Tuple

from PIL.ImageFont import FreeTypeFont

from core.element.BaseImg import BaseImg
from core.element.CharImg import CharImg
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
import json
from utils import time_util as tu
import math
import traceback
from utils import log

TYPE_ORIENTATION_HORIZONTAL = 0
TYPE_ORIENTATION_VERTICAL = 1

TYPE_ALIGN_MODEL_B = 0  # Text alignment mode: bottom/left alignment
TYPE_ALIGN_MODEL_C = 1  # Text alignment mode: center alignment
TYPE_ALIGN_MODEL_T = 2  # Text alignment mode: top/right alignment

class FreeTypeFontOffset(FreeTypeFont):
    """
    Custom Font Type for hacking around the offset issue
    This was done here for very specific use case so check before using it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getsize(self, text, direction=None, features=None, language=None, stroke_width=0) -> Tuple[int, int]:
        size = super().getsize(text)
        offset = self.getoffset(text)
        stroke_width = 0
        size = (size[0] + stroke_width * 2, size[1] + stroke_width * 2 - offset[1],)  # hack
        return size

    def getmask2(
            self,
            text,
            mode="",
            fill=Image.core.fill,
            direction=None,
            features=None,
            language=None,
            stroke_width=0,
            anchor=None,
            ink=0,
            *args,
            **kwargs,
    ):
        """
        Create a bitmap for the text.
        """
        size, offset = self.font.getsize(
            text, mode, direction, features, language, anchor
        )
        offset = (0, 0)  # Hack
        size = size[0] + stroke_width * 2, size[1] + stroke_width * 2
        offset = offset[0] - stroke_width, offset[1] - stroke_width
        Image._decompression_bomb_check(size)
        im = fill("RGBA" if mode == "RGBA" else "L", size, 0)
        self.font.render(
            text, im.id, mode, direction, features, language, stroke_width, ink
        )
        return im, offset

    def getbbox(
            self,
            text,
            mode="",
            direction=None,
            features=None,
            language=None,
            stroke_width=0,
            anchor=None,
    ):
        size = self.getsize(text)
        offset = (0, 0)  # hack
        stroke_width = 0
        left, top = offset[0] - stroke_width, offset[1] - stroke_width
        width, height = size[0] + 2 * stroke_width, size[1] + 2 * stroke_width
        return left, top, left + width, top + height


class TextImg(BaseImg):
    """
    String picture object
    """

    def __init__(self,
                 char_obj_list: List[CharImg],
                 text_img_output_dir,
                 text_img_info_output_dir,
                 orientation,
                 align_mode,
                 img: Image.Image = None,
                 img_path: str = None,
                 **kwargs
                 ):
        tmp_list = []
        for item in char_obj_list:
            if isinstance(item, dict):
                tmp_list.append(CharImg(**item))
        if tmp_list:
            char_obj_list = tmp_list

        self.char_obj_list = char_obj_list
        self.text = "".join([char_obj.char for char_obj in self.char_obj_list])
        self.text_img_output_dir = text_img_output_dir
        self.text_img_info_output_dir = text_img_info_output_dir
        self.orientation = orientation
        self.align_mode = align_mode

        if img_path:
            self.img_name = img_path.split(os.sep)[-1]
            self.name = self.img_name.split('.')[0]
            self.img_path = img_path
            self.img = load_img(self.img_path)
        else:
            self.name = self._gen_name(align_mode, orientation)
            self.img_name = self.name + ".png"
            self.img_path = os.path.join(text_img_output_dir, self.img_name)
            self.img = img

    def _gen_name(self, align_mode, orientation):
        o = "v" if orientation == TYPE_ORIENTATION_VERTICAL else "h"
        a = 'b'
        if align_mode == TYPE_ALIGN_MODEL_T:
            a = 't'
        elif align_mode == TYPE_ALIGN_MODEL_C:
            a = 'c'
        return tu.timestamp() + "_" + o + "_" + a + "_" + self.text.replace(" ", "_")

    def __repr__(self):
        return json.dumps(self.__dict__, cls=CharImgEncoder)

    def export(self):
        """
        Data output
        :return:
        """
        self.img.save(self.img_path)
        json_file_path = os.path.join(self.text_img_info_output_dir, self.name + ".json")
        with open(json_file_path, 'w') as f:
            json.dump(self.__dict__, f, cls=CharImgEncoder)

    @staticmethod
    def load_from_json(file_path):
        """
        Load objects from json file
        :param file_path:
        :return:
        """
        assert os.path.exists(file_path), "json file is not exist,please check: {file_path}".format(file_path=file_path)
        with open(file_path, 'r') as f:
            j = json.load(f)
            return TextImg(**j)

    def show(self, with_box=False):
        """
        Show pictures
        :param with_box:
        :return:
        """
        image = self.cv_img()

        if with_box:
            for char_obj in self.char_obj_list:
                pt1 = (char_obj.box[0], char_obj.box[1])
                pt2 = (char_obj.box[2], char_obj.box[3])
                image = cv2.rectangle(image, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)

        cv2.imshow(self.text, image)
        cv2.waitKey()
        cv2.destroyWindow(self.text)

    def cv_img(self):
        """
        Get the image object of opencv
        :return:
        """
        image = np.array(self.img)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        return image

    def pil_img(self):
        """
        Get the image object of pillow
        :return:
        """
        return self.img


class CharImgEncoder(json.JSONEncoder):
    def default(self, o):
        if not isinstance(o, Image.Image):
            return o.__dict__


def load_img(img_path):
    """
    Load image files from disk
    :param img_path:
    :return:
    """
    assert os.path.exists(img_path), "image is not exist, please check. {img_path}".format(img_path=img_path)
    return Image.open(img_path)


def calc_bg_size(font_path: str,
                 orientation: int,
                 char_obj_list: List[CharImg],
                 spacing_rate: float,
                 padding,
                 auto_padding_to_ratio) -> tuple:
    """
    Calculate background size
    :param font_path: Font path
    :param orientation:Towards
    :param char_obj_list: Character object
    :param spacing_rate: Spacing (as a percentage of text size)
    :param padding: Inner margin
    :param auto_padding_to_ratio: Automatic padding to the specified ratio (horizontal arrangement is w/h, vertical arrangement is h/w)
    :return:
    """

    max_char_bg_w = 0
    max_char_bg_h = 0

    bg_w = 0
    bg_h = 0

    for index, char_obj in enumerate(char_obj_list):
        # font = ImageFont.truetype(font_path, size=char_obj.font_size)
        font = FreeTypeFontOffset(font_path, size=char_obj.font_size)

        char_bg_w = 0
        char_bg_h = 0
        try:
            char_bg_w, char_bg_h = font.getsize(char_obj.char)
            bbox = font.getbbox(char_obj.char)

            # print(f'char : {char_obj.char} , {char_bg_h} x {char_bg_w} : {bbox}')

            # Add border size
            char_bg_w += char_obj.border_width * 2
            char_bg_h += char_obj.border_width * 2
        except Exception as e:
            traceback.print_exc()
        char_obj.size = (char_bg_w, char_bg_h)

        # Get the width and height of the largest character image in the current line of text
        max_char_bg_w = char_bg_w if char_bg_w > max_char_bg_w else max_char_bg_w
        max_char_bg_h = char_bg_h if char_bg_h > max_char_bg_h else max_char_bg_h

        # Determine whether the position of the last character has been traversed
        is_last = index == len(char_obj_list) - 1

        r = 0 if is_last else spacing_rate

        if orientation == TYPE_ORIENTATION_VERTICAL:
            bg_w = max_char_bg_w
            bg_h += math.ceil(char_obj.size[1] * (1 + r))
        else:
            bg_w += math.ceil(char_obj.size[0] * (1 + r))
            bg_h = max_char_bg_h

    if auto_padding_to_ratio > 0:
        # Automatic padding to the specified size

        # If it is arranged horizontally, add padding on the left and right sides
        # auto_padding_to_ratio = tw / th
        if orientation == TYPE_ORIENTATION_HORIZONTAL:
            st_w = auto_padding_to_ratio * bg_h
            if st_w > bg_w:
                d = round((st_w - bg_w) / 2)
                padding = (d, 0, d, 0)
            else:
                st_h = bg_w / auto_padding_to_ratio
                d = round((st_h - bg_h) / 2)
                padding = (0, d, 0, d)

        # If it is arranged vertically, add padding on the upper and lower sides
        # auto_padding_to_ratio = th / tw
        elif orientation == TYPE_ORIENTATION_VERTICAL:
            st_h = auto_padding_to_ratio * bg_w
            if st_h > bg_h:
                d = round((st_h - bg_h) / 2)
                padding = (0, d, 0, d)
            else:
                st_w = bg_h / auto_padding_to_ratio
                d = round((st_w - bg_w) / 2)
                padding = (d, 0, d, 0)

    bg_w = bg_w + padding[0] + padding[2]
    bg_h = bg_h + padding[1] + padding[3]

    return bg_w, bg_h, padding


def draw_text(font_path, bg_w, bg_h, orientation, char_obj_list: List[CharImg], spacing_rate, align_mode, padding):
    """
    Draw text on the text map background
    :param font_path:
    :param bg_w:
    :param bg_h:
    :param orientation:
    :param char_obj_list:
    :param spacing_rate:
    :param align_mode:
    :param padding:
    :return:
    """
    img = Image.new("RGBA", (bg_w, bg_h), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_area_w = bg_w - padding[0] - padding[2]
    font_area_h = bg_h - padding[1] - padding[3]

    tmp_char = None
    l, t = 0, 0
    for index, char_obj in enumerate(char_obj_list):
        # font = ImageFont.truetype(font_path, size=char_obj.font_size)
        font = FreeTypeFontOffset(font_path, size=char_obj.font_size)

        cw, ch = char_obj.size

        if orientation == TYPE_ORIENTATION_VERTICAL:
            if align_mode == TYPE_ALIGN_MODEL_B:
                l = 0
            elif align_mode == TYPE_ALIGN_MODEL_C:
                l = math.ceil((font_area_w - cw) / 2)
            elif align_mode == TYPE_ALIGN_MODEL_T:
                l = font_area_w - cw

            if tmp_char:
                add_t = math.ceil(tmp_char.size[1] * (1 + spacing_rate))
                t += add_t
            else:
                t = 0

            l += padding[0]
            if index == 0:
                t += padding[1]
            char_obj.box = [l, t, l + cw, t + ch]

        else:
            t = 0
            if align_mode == TYPE_ALIGN_MODEL_B:
                t = font_area_h - ch
            elif align_mode == TYPE_ALIGN_MODEL_C:
                t = math.ceil((font_area_h - ch) / 2)
            elif align_mode == TYPE_ALIGN_MODEL_T:
                t = 0

            if tmp_char:
                add_l = math.ceil(tmp_char.size[0] * (1 + spacing_rate))
                l += add_l
            else:
                l = 0

            t += padding[1]
            if index == 0:
                l += padding[0]
            char_obj.box = [l, t, l + cw, t + ch]

        # log.info("draw text >> {text} color: {color} box: {box}".format(text=char_obj.char,
        #                                                                 color=char_obj.color,
        #                                                                 box=char_obj.box))
        draw.text((l + char_obj.border_width, t + char_obj.border_width),
                  text=char_obj.char,
                  fill=char_obj.color,
                  font=font)
        if char_obj.border_width > 0:
            draw.rectangle(xy=tuple(char_obj.box), width=char_obj.border_width, outline=char_obj.border_color)
        tmp_char = char_obj
    return img


def gen_batch_char_obj(text,
                       color,
                       font_size,
                       border_width=0,
                       border_color=(0, 0, 0, 0)) -> List[CharImg]:
    """
    Generate a batch of CharImg objects
    :param text:
    :param color:
    :param font_size:
    :param border_width:
    :param border_color:
    :return:
    """
    char_obj_list = []
    for char in text:
        char_obj_list.append(
            CharImg(char, font_size=font_size, color=color, border_width=border_width, border_color=border_color))
    return char_obj_list


def create(char_obj_list: List[CharImg],
           orientation: int = TYPE_ORIENTATION_HORIZONTAL,
           align_mode: int = TYPE_ALIGN_MODEL_B,
           spacing_rate: float = 0.08,
           padding=(0, 0, 0, 0),
           auto_padding_to_ratio=0,
           font_path="",
           text_img_output_dir="",
           text_img_info_output_dir=""
           ):
    """
    Generate text image
    :param char_obj_list: Character object list
    :param orientation: Direction of generation
    :param align_mode: Text alignment mode
    :param spacing_rate: Spacing (as a percentage of text size)
    :param padding: Inner margin
    :param auto_padding_to_ratio: Automatic padding to the specified ratio <=0 means no automatic padding (horizontal arrangement is w/h, vertical arrangement is h/w)
    :param font_path: Font file path
    :param text_img_output_dir:
    :param text_img_info_output_dir:
    :return:
    """
    bg_w, bg_h, padding = calc_bg_size(font_path, orientation, char_obj_list, spacing_rate, padding,
                                       auto_padding_to_ratio)

    img = draw_text(font_path, bg_w, bg_h, orientation, char_obj_list, spacing_rate, align_mode, padding)

    return TextImg(char_obj_list=char_obj_list,
                   text_img_output_dir=text_img_output_dir,
                   text_img_info_output_dir=text_img_info_output_dir,
                   orientation=orientation,
                   align_mode=align_mode,
                   img=img)
