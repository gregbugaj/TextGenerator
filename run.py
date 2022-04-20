import numpy as np
from PIL import ImageDraw, ImageFont
from PIL import Image

import service
import argparse
import os
import cv2

from core.element.TextImg import FreeTypeFontOffset
from service import SmoothAreaProvider


def image_resizeXXXX(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_and_frameXXX(image, width, height, color=255):
    ## Merge two images
    img = image_resize(image, height=height)
    l_img = np.ones((height, height, 3), np.uint8) * color
    s_img = img  # np.zeros((512, 512, 3), np.uint8)
    x_offset = int((l_img.shape[1] - img.shape[1]) / 2)
    y_offset = int((l_img.shape[0] - img.shape[0]) / 2)
    l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    return l_img

def resize_image(image, desired_size, color=(255, 255, 255)):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------
    
    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    '''

    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image


def directory_resize(dir_src, dir_dest, desired_size):
    print(dir_src)
    print(dir_dest)

    filenames = os.listdir(dir_src)
    filenames.sort()
    if not os.path.exists(dir_dest):
        os.makedirs(dir_dest)

    for filename in filenames:
        try:
            print(filename)
            # open image file
            path = os.path.join(dir_src, filename)
            path_dest = os.path.join(dir_dest, filename) + "_.png"

            img = cv2.imread(path)
            img = resize_image(img, desired_size)
            img_h, img_w, img_c = img.shape
            print(img.shape)

            cv2.imwrite(path_dest, img)
        except Exception as e:
            print(e)

def get_alpha_mask(img):
    r, g, b, a = img.split()
    return a

def get_rotate_box(img):
    alpha = get_alpha_mask(img)
    alpha = np.asarray(alpha, np.uint8)

    points = np.argwhere(alpha > 0)
    points = points[:, ::-1]
    print(len(points))
    rotate_rect = cv2.minAreaRect(points)
    rotate_rect_point = cv2.boxPoints(rotate_rect)
    rotate_rect_point = rotate_rect_point.astype(np.int32)
    return rotate_rect_point

def test_textsize_equal():
    font_size = 85
    font_path = './assets/font/FreeMono.ttf'

    im = Image.new(mode='RGBA', size=(300, 100), color=(0, 0, 0))
    # im = Image.new(mode='RGBA', size=(300, 100), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    ttf = FreeTypeFontOffset(font_path, size=font_size)

    txt = "7"
    size = draw.textsize(txt, ttf)
    xy = (10, 10)
    draw.text(xy, txt, font=ttf)
    # draw.rectangle((xy[0], xy[1], xy[0]+size[0], xy[1]+size[1]))
    bbox = ttf.getbbox(txt)

    print(size)
    print(bbox)
    im.save('/home/greg/dev/TextGenerator/rectangle_surrounding_text.png')

    image = cv2.imread('/home/greg/dev/TextGenerator/rectangle_surrounding_text.png')
    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = img_gray
    # ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    print(f'contours len : {len(contours)}')
    print(len(contours[0]))
    # draw contours on the original image
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                     lineType=cv2.LINE_AA)

    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('contours_none_image1.png', image_copy)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    if False:
        smooth = SmoothAreaProvider(down_scale=64,
                     anchor_ratio=(0.17, 0.25, 0.5, 1.0, 2.0, 3.0, 4),
                     anchor_scale=(4, 8, 16, 24, 32, 48, 64, 72)
        )
        image = cv2.imread("/home/gbugaj/devio/TextGenerator/assets/img/001.png")
        rects = smooth.get_image_rects(image)
        print(rects)
        for rect in rects:
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (120, 78, 255), 2)
        cv2.imwrite('test.jpg', image)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)

    # test_textsize_equal()
    service.start()

    # src = '/home/greg/dev/TextGenerator/output_BOX_33_TEST/icdar_data'
    # dst = '/home/greg/dev/TextGenerator/output_BOX_33_TEST/resized_1024/image'

    # directory_resize(src, dst, (1024, 1024))

    # src = '/home/greg/dev/TextGenerator/output_BOX_33_TEST/mask_data'
    # dst = '/home/greg/dev/TextGenerator/output_BOX_33_TEST/resized_1024/mask'
    # directory_resize(src, dst, (1024, 1024))

    # src = '/home/greg/dev/TextGenerator/output/icdar_data'
    # dst = '/home/greg/dev/TextGenerator/output/resized_1024/image'

    # directory_resize(src, dst, (1024, 1024))

    # src = '/home/greg/dev/TextGenerator/output/mask_data'
    # dst = '/home/greg/dev/TextGenerator/output/resized_1024/mask'
    # directory_resize(src, dst, (1024, 1024))
