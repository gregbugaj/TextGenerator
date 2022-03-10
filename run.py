import service
import argparse
import os
import cv2

def image_resizeXXXX(image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def resize_and_frameXXX(image, width, height, color = 255):
    ## Merge two images
    img = image_resize(image, height=height)
    l_img = np.ones((height, height, 3), np.uint8) * color
    s_img = img # np.zeros((512, 512, 3), np.uint8)
    x_offset = int((l_img.shape[1] - img.shape[1]) / 2) 
    y_offset = int((l_img.shape[0] - img.shape[0]) / 2)
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

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
        ratio_w = float(desired_size[0])/size[0]
        ratio_h = float(desired_size[1])/size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x*ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image



def directory_resize(dir_src, dir_dest,desired_size):
    print(dir_src)
    print(dir_dest)

    filenames = os.listdir(dir_src)
    filenames.sort()
    if not os.path.exists(dir_dest):
        os.makedirs(dir_dest)

    for filename in filenames:
        try:
            print (filename)
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

if __name__ == '__main__':
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


