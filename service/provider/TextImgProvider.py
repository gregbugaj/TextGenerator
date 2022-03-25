import os
import time
from typing import List
from constant import const
from utils.decorator import singleton
from utils.random_tools import Random
from core.element.CharImg import CharImg
from core.element.TextImg import create, gen_batch_char_obj, TYPE_ORIENTATION_HORIZONTAL, TYPE_ORIENTATION_VERTICAL, \
    TYPE_ALIGN_MODEL_C, TYPE_ALIGN_MODEL_B, TYPE_ALIGN_MODEL_T
from core.layout.strategy.HorizontalStrategy import HorizontalStrategy
from core.layout.strategy.VerticalStrategy import VerticalStrategy
from core.layout.strategy.HorizontalFlowStrategy import HorizontalFlowStrategy
from core.layout.strategy.VerticalFlowStrategy import VerticalFlowStrategy
from core.layout.strategy.CustomizationStrategy1 import CustomizationStrategy1
from core.layout import TextBlock, NextBlockGenerator
from utils import font_tool
import numpy as np
import cv2


def list_font_path(font_file_dir):
    """
    :param font_file_dir: Font file storage path
    :return:
    """
    assert os.path.exists(font_file_dir), "font_file_dir is not exist, please check: {font_file_dir}".format(
        font_file_dir=font_file_dir)
    path_list = []
    for item in os.listdir(font_file_dir):
        path = os.path.join(font_file_dir, item)
        path_list.append(path)
    return path_list


@singleton
class TextImgProvider(NextBlockGenerator):

    def __init__(self, font_file_dir, text_img_output_dir, text_img_info_output_dir, font_min_size, font_max_size,
                 use_char_common_color_probability,
                 char_common_color_list,
                 char_border_width,
                 char_border_color,
                 auto_padding_to_ratio=0.0,
                 seed=time.time()):
        """
        Initialize the text image generator
        :param font_file_dir: Font file directory
        :param text_img_output_dir: Text and image output directory
        :param text_img_info_output_dir: Text and image data output directory
        :param font_min_size: The minimum text font size
        :param use_char_common_color_probability
        :param char_common_color_list
        :param char_border_width: The width of the character border
        :param char_border_color: The color of the character border
        :param auto_padding_to_ratio: Automatic padding to the specified ratio <=0 means no automatic padding (horizontal arrangement is w/h, vertical arrangement is h/w)
        :param seed:
        """
        os.makedirs(text_img_output_dir, exist_ok=True)
        os.makedirs(text_img_info_output_dir, exist_ok=True)

        if not seed:
            seed = time.time()

        self.font_file_list = list_font_path(font_file_dir)
        self._font_index = 0
        self.text_img_output_dir = text_img_output_dir
        self.text_img_info_output_dir = text_img_info_output_dir
        self.font_min_size = font_min_size
        self.font_max_size = font_max_size
        self.use_char_common_color_probability = use_char_common_color_probability
        self.char_common_color_list = char_common_color_list
        self.char_border_width = char_border_width
        self.char_border_color = eval(char_border_color) if type(char_border_color) is str else char_border_color
        self.auto_padding_to_ratio = auto_padding_to_ratio

        Random.shuffle(self.font_file_list, seed)

    def next_font_path(self):
        """
        :return:
        """
        font_path = self.font_file_list[self._font_index]
        self._font_index += 1
        if self._font_index >= len(self.font_file_list):
            self._font_index = 0
        return font_path

    def gen_text_img(self, text: str,
                     font_path,
                     color=const.COLOR_BLACK,
                     font_size=14,
                     border_width=0,
                     border_color=const.COLOR_TRANSPARENT,
                     orientation=TYPE_ORIENTATION_HORIZONTAL,
                     padding=(0, 0, 0, 0),
                     align_mode=TYPE_ALIGN_MODEL_C,
                     auto_padding_to_ratio=0.0):
        char_obj_list = gen_batch_char_obj(text=text, color=color, font_size=font_size, border_width=border_width,
                                           border_color=border_color)

        text_img = create(char_obj_list=char_obj_list,
                          orientation=orientation,
                          align_mode=align_mode,
                          padding=padding,
                          auto_padding_to_ratio=auto_padding_to_ratio,
                          font_path=font_path,
                          text_img_output_dir=self.text_img_output_dir,
                          text_img_info_output_dir=self.text_img_info_output_dir)
        return text_img

    def gen_complex_text_img(self, char_obj_list: List[CharImg],
                             font_path,
                             orientation=TYPE_ORIENTATION_HORIZONTAL,
                             align_mode=TYPE_ALIGN_MODEL_C):
        """
        :param char_obj_list:
        :param font_path:
        :param orientation:
        :param align_mode:
        :return:
        """
        text_img = create(char_obj_list=char_obj_list,
                          orientation=orientation,
                          align_mode=align_mode,
                          font_path=font_path,
                          text_img_output_dir=self.text_img_output_dir,
                          text_img_info_output_dir=self.text_img_info_output_dir)
        return text_img

    def get_fontcolor(self, bg_img):
        """
        get font color by mean
        :param bg_img:
        :return:
        """
        char_common_color_list = self.char_common_color_list

        if Random.random_float(0, 1) <= self.use_char_common_color_probability and char_common_color_list:
            return eval(Random.random_choice_list(char_common_color_list))
        else:
            image = np.asarray(bg_img)
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

            bg = lab_image[:, :, 0]
            l_mean = np.mean(bg)

            new_l = Random.random_int(0, 127 - 80) if l_mean > 127 else Random.random_int(127 + 80, 255)
            new_a = Random.random_int(0, 255)
            new_b = Random.random_int(0, 255)

            lab_rgb = np.asarray([[[new_l, new_a, new_b]]], np.uint8)
            rbg = cv2.cvtColor(lab_rgb, cv2.COLOR_Lab2RGB)

            r = rbg[0, 0, 0]
            g = rbg[0, 0, 1]
            b = rbg[0, 0, 2]

            return (r, g, b, 255)

    def auto_gen_next_img(self, width, height, strategy, bg_img, block_list):
        """
        Automatically generate the next text map
        :return:
        """
        from service import text_provider
        import string
        import random
        text = "".join(text_provider.gen.__next__())

        # GB : Mod
        if np.random.choice([0, 1], p=[0.5, 0.5]):
            letters = string.digits
            c = np.random.randint(1, 9)
            text = (''.join(random.choice(letters) for i in range(c)))

        fp = self.next_font_path()

        if isinstance(strategy, HorizontalStrategy):
            orientation = TYPE_ORIENTATION_HORIZONTAL
        elif isinstance(strategy, VerticalStrategy):
            orientation = TYPE_ORIENTATION_HORIZONTAL
        elif isinstance(strategy, HorizontalFlowStrategy):
            orientation = TYPE_ORIENTATION_HORIZONTAL
        elif isinstance(strategy, VerticalFlowStrategy):
            orientation = TYPE_ORIENTATION_VERTICAL
        elif isinstance(strategy, CustomizationStrategy1):
            if block_list:
                orientation = TYPE_ORIENTATION_HORIZONTAL
            else:
                orientation = TYPE_ORIENTATION_VERTICAL
        else:
            orientation = Random.random_choice_list(
                [TYPE_ORIENTATION_VERTICAL, TYPE_ORIENTATION_HORIZONTAL, TYPE_ORIENTATION_HORIZONTAL])

        if self.font_max_size != 'vaild':
            font_size = Random.random_int(self.font_min_size, self.font_max_size)
        else:
            v = min(width, height)
            font_size = Random.random_int(v // 20, v // 10)
            font_size = self.font_min_size if font_size < self.font_min_size else font_size

        # Eliminate non-existent text
        text = "".join(filter(lambda c: font_tool.check(c, font_path=fp), text))
        if len(text) >= 2:
            # Generate text image
            align = Random.random_choice_list(
                [TYPE_ALIGN_MODEL_B, TYPE_ALIGN_MODEL_T, TYPE_ALIGN_MODEL_C])
            # text = "XYZ"
            # font_size = 32
            align = 0
            text_img = self.gen_text_img(text,
                                         font_size=font_size,
                                         border_width=self.char_border_width,
                                         border_color=self.char_border_color,
                                         color=self.get_fontcolor(bg_img),
                                         orientation=orientation,
                                         align_mode=align,
                                         font_path=fp,
                                         auto_padding_to_ratio=self.auto_padding_to_ratio)
            return text_img

    def auto_gen_next_img_block(self, width, height, strategy, bg_img, block_list, rotate_angle):
        next_img = self.auto_gen_next_img(width=width,
                                          height=height,
                                          strategy=strategy,
                                          bg_img=bg_img,
                                          block_list=block_list)
        if next_img:
            return TextBlock(text_img=next_img, margin=10, rotate_angle=rotate_angle)


if __name__ == '__main__':
    from service import init_config

    init_config()
    from service import text_img_provider

    fp = text_img_provider.next_font_path()

    p = text_img_provider.gen_text_img("hello world", color=const.COLOR_BLUE, font_path=fp)
    p.export()
    # p.show()

    l = []
    l.extend(gen_batch_char_obj("你好啊", const.COLOR_BLUE, font_size=24))
    l.extend(gen_batch_char_obj(" 渣 男 ", const.COLOR_GREEN, font_size=28))
    r = text_img_provider.gen_complex_text_img(l, font_path=fp)
    r.show()

    bg_w, bg_h = text_img_provider.calc_bg_size(fp, orientation=TYPE_ORIENTATION_HORIZONTAL, char_obj_list=l,
                                                spacing_rate=0.1)

    print(bg_w)
    print(bg_h)
