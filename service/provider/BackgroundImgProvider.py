"""
Name : BackgroundImgProvider.py
Author  : Hanat
Time    : 2019-09-19 16:50
Desc:
"""
import os
import cv2
import math
import numpy as np
from utils.random_tools import Random
from PIL import Image


class DirImageGen(object):

    def __init__(self, image_list: list, len_range=(2, 15)):
        """

        :param character_seq:
        :param batch_size:
        """        
        self.augmentor = BackgroundImageAugmentor()
        self._image_list = image_list
        # random.shuffle(self.character_seq)
        self._len_range = len_range
        self.get_next = self._get_next_image()
        self._imgs_length = len(self._image_list)

    def _get_next_image(self):
        seek = 0
        while True:
            if seek == self._imgs_length:
                seek = 0
            print("test:", self._image_list[seek])
            yield self.augmentor.augment(cv2.imread(self._image_list[seek]))
            seek += 1


class GenrateGaussImage(object):

    def __init__(self, width_range=(1000, 1500), height_range=(1000, 1500)):
        """

        :param width_range: generate image width size range
        :param height_range: generate image height size range
        """
        self.augmentor = BackgroundImageAugmentor()
        self._width_range = width_range
        self._height_range = height_range
        self.get_next = self.get_gauss_image()

    def get_gauss_image(self):
        while True:
            bg = self.apply_gauss_blur()
            yield cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    def apply_gauss_blur(self, ks=None):
        """

        :param ks: guass kernal window size
        :return:
        """
        bg_high = Random.random_float(220, 255)
        bg_low = bg_high - Random.random_float(1, 60)
        width = Random.random_int(self._width_range[0], self._width_range[1])
        height = Random.random_int(self._height_range[0], self._height_range[1])
        img = np.random.randint(bg_low, bg_high, (height, width)).astype(np.uint8)
        if ks is None:
            ks = [7, 9, 11, 13]
        k_size = Random.random_choice_list(ks)
        sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
        sigma = 0
        if k_size <= 3:
            sigma = Random.random_choice_list(sigmas)
        img = cv2.GaussianBlur(img, (k_size, k_size), sigma)
        img = self.augmentor.augment(img)
        return img


class GenrateQuasicrystalImage(object):

    def __init__(self, width_range=(1000, 1500), height_range=(1000, 1500)):
        """

        :param width_range: generate image width size range
        :param height_range: generate image height size range
        """
        self._width_range = width_range
        self._height_range = height_range
        self.get_next = self.get_quasicrystal_image()

    def get_quasicrystal_image(self):
        while True:
            bg = self.apply_quasicrystal()
            yield bg

    def apply_quasicrystal(self):
        """
            Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
        """
        width = Random.random_int(self._width_range[0], self._width_range[1])
        height = Random.random_int(self._height_range[0], self._height_range[1])

        image = np.zeros((height, width, 3), dtype=np.uint8)
        rotation_count = Random.random_int(10, 20)
        y_vec = np.arange(start=0, stop=width, dtype=np.float32)
        x_vec = np.arange(start=0, stop=height, dtype=np.float32)

        grid = np.meshgrid(y_vec, x_vec)
        y_matrix = np.reshape(np.asarray(grid[0]) / (width - 1) * 4 * math.pi - 2 * math.pi, (height, width, 1))
        x_matrix = np.reshape(np.asarray(grid[1]) / (height - 1) * 4 * math.pi - 2 * math.pi, (height, width, 1))
        y_matrix_3d = np.repeat(y_matrix, rotation_count, axis=-1)
        x_matrix_3d = np.repeat(x_matrix, rotation_count, axis=-1)

        rotation_vec = np.arange(start=0, stop=rotation_count, dtype=np.float32)
        rotation_vec = np.reshape(rotation_vec, newshape=(1, 1, rotation_count))

        for k in range(3):
            frequency = Random.random_float(0, 1) * 30 + 20  # frequency
            phase = Random.random_float(0, 1) * 2 * math.pi  # phase

            r = np.hypot(x_matrix_3d, y_matrix_3d)
            a = np.arctan2(y_matrix_3d, x_matrix_3d) + (rotation_vec * math.pi * 2.0 / rotation_count)
            z = np.cos(r * np.sin(a) * frequency + phase)
            z = np.sum(z, axis=-1)

            c = 255 - np.round(255 * z / rotation_count)
            c = np.asarray(c, dtype=np.uint8)
            image[:, :, k] = c

        return image


class BackgroundImgProvider(object):

    def __init__(self, bg_img_conf):
        self.gen = self.get_generator(bg_img_conf)

    def get_generator(self, bg_img_conf):
        """
        generator
        :param bg_img_conf:
        :return:
        """
        gen_probability = []
        all_generator = []
        for item in bg_img_conf:
            t = item['type']
            probability = float(item['probability'])
            if t == 'from_dir':
                bg_img_dir = item['dir']
                img_path_list = [os.path.join(bg_img_dir, img) for img in os.listdir(bg_img_dir) if ('.DS' not in img)]
                dir_img_gen = DirImageGen(img_path_list)
                all_generator.append(dir_img_gen)
            elif t == 'from_generate':
                width_range = eval(item['width_range'])
                height_range = eval(item['height_range'])
                gauss_img_gen = GenrateGaussImage(width_range=width_range, height_range=height_range)
                all_generator.append(gauss_img_gen)
            gen_probability.append(probability)

        value_list = gen_probability
        len_gen = len(all_generator)
        while True:
            index = Random.random_choice(list(value_list))
            if index <= len_gen and all_generator[index]:
                np_img = all_generator[index].get_next.__next__()
                np_img = np_img[..., ::-1]
                img = Image.fromarray(np_img, mode='RGB')
                yield img

    def generator(self):
        return self.gen.__next__()

class BackgroundImageAugmentor(object):
    def __init__(self) -> None:
        super().__init__()

    def augment(self, img):

        # if True:
        #     return img
        noisy_img = img.copy()
        h = noisy_img.shape[0]
        w = noisy_img.shape[1]

        # noise = cv2.threshold(noisy_img, 128, 255, cv2.THRESH_BINARY)[1]
        noise = noisy_img
        # return noise
        print('Augmenting image : {}'.format(noisy_img.shape))

        vertical_bool = {'left': np.random.choice([0,1], p =[0.3, 0.7]), 'right': np.random.choice([0,1])} # [1 or 0, 1 or 0] whether to make vertical left line on left and right side of the image
        for left_right, bool_ in vertical_bool.items():
            if bool_:
                print('left_right: ', left_right)
                if left_right == 'left':
                    v_start_x = np.random.randint(5, int(noisy_img.shape[1]*0.06))
                else:
                    v_start_x = np.random.randint(int(noisy_img.shape[1]*0.95), noisy_img.shape[1] - 5)

                v_start_y = np.random.randint(0, int(noisy_img.shape[0]*0.06))
                v_end_y   = np.random.randint(int(noisy_img.shape[0]*0.95), noisy_img.shape[0])

                y_points = list(range(v_start_y, v_end_y + 1))
                y_points_black_prob = np.random.choice([0,1], size = len(y_points), p = [0.2, 0.8])

                for idx, y in enumerate(y_points):
                    if y_points_black_prob[idx]:
                        noisy_img[y, v_start_x - np.random.randint(2): v_start_x + np.random.randint(2)] = np.random.randint(0,30)

        text_height=12
        y_line_list=[]

        y_line_count = np.random.randint(1, 8)
        for y in range(0, y_line_count):
            y_line_list.append(np.random.randint(noisy_img.shape[0]))

        if True or np.random.choice([True, False], p = [0.60, 0.40]):
            # adding horizontal line (noise)
            for y_line in y_line_list: 
                # samples the possibility of adding a horizontal line
                add_horizontal_line = np.random.choice([0, 1], p = [0.5, 0.5])
                if not add_horizontal_line:
                    continue

                # shift y_line randomly in the y-axis within a defined limit
                limit = int(text_height*0.3)
                if limit == 0: # this happens when the text used for getting the text height is '-', ',', '=' and other little symbols like these 
                    limit = 10
                y_line += np.random.randint(-limit, limit)

                h_start_x = np.random.randint(0, noisy_img.shape[1])                           # min x of the horizontal line
                h_end_x   = np.random.randint(int(noisy_img.shape[1]*0.8), noisy_img.shape[1]) # max x of the horizontal line
                h_length = h_end_x - h_start_x + 1
                num_h_lines = np.random.randint(10,30) # partitions to be made in the horizontal line (necessary to make it look like naturally broken lines)
                h_lines = []
                h_start_temp = h_start_x
                next_line = True

                num_line = 0
                while (next_line) and (num_line < num_h_lines):
                    if h_start_temp < h_end_x:
                        h_end_temp = np.random.randint(h_start_temp + 1, h_end_x + 1)
                        if h_end_temp < h_end_x:
                            h_lines.append([h_start_temp, h_end_temp]) 
                            h_start_temp = h_end_temp + 1
                            num_line += 1
                        else:
                            h_lines.append([h_start_temp, h_end_x]) 
                            num_line += 1
                            next_line = False
                    else:
                        next_line = False

                for h_line in h_lines:
                    use_solid_line = np.random.choice([0, 1], p = [0.5, 0.5])
                    col = np.random.choice(['black', 'white'], p = [0.95, 0.05]) # probabilities of line segment being a solid one or a broken one
                    if True or col == 'black':
                        x_points = list(range(h_line[0], h_line[1] + 1))
                        x_points_black_prob = np.random.choice([0,1], size = len(x_points), p = [0, 1])

                        for idx, x in enumerate(x_points):
                            if True or x_points_black_prob[idx]:
                                if use_solid_line:                    
                                    noisy_img[y_line : y_line + 1, x] = np.random.randint(0,30)  
                                else:
                                    noisy_img[y_line - np.random.randint(2): y_line + np.random.randint(2), x] = np.random.randint(0, 30)  

        noise = cv2.threshold(noisy_img, 128, 255, cv2.THRESH_BINARY)[1]
        return noise