from lxml.etree import Element, SubElement, tostring
from utils import log
import shutil
import json
import os
import cv2
import numpy as np
import hashlib

IMAGE_INDEX = 0
ANNOTATION_INDEX = 0

def get_pic_dir(out_put_dir):
    img_dir = os.path.join(out_put_dir, "img")
    pic_dir = os.path.join(img_dir, "pic")
    return pic_dir


def get_fragment_dir(out_put_dir):
    img_dir = os.path.join(out_put_dir, "img")
    fragment_dir = os.path.join(img_dir, "fragment")
    return fragment_dir


def get_data_dir(out_put_dir):
    data_dir = os.path.join(out_put_dir, "data")
    return data_dir


def get_label_data_dir(out_put_dir):
    label_data = os.path.join(out_put_dir, "label_data")
    return label_data


def get_voc_data_dir(out_put_dir):
    voc_data = os.path.join(out_put_dir, "voc_data")
    return voc_data


def get_coco_data_dir(out_put_dir):
    coco_data = os.path.join(out_put_dir, "coco_data")
    return coco_data


def get_lsvt_data_dir(out_put_dir):
    lsvt_data = os.path.join(out_put_dir, "lsvt_data")
    return lsvt_data


def get_icidar_data_dir(out_put_dir):
    icdr_data = os.path.join(out_put_dir, "icdar_data")
    return icdr_data


def get_mask_data_dir(out_put_dir):
    mask_data = os.path.join(out_put_dir, "mask_data")
    return mask_data


def gen_all_pic():
    """
    Generate all pictures
    :return:
    """
    from service import conf
    gen_count = conf['base']['count_per_process']
    index = 0

    global IMAGE_INDEX
    while index < gen_count:
        log.info("-" * 20 + " generate new picture {index}/{gen_count}".format(index=index,
                                                                               gen_count=gen_count) + "-" * 20)
        dump_data = gen_pic()
        # Write label
        if dump_data:
            add_label_data(dump_data)
            # Write voc
            if conf['base']['gen_voc']:
                gen_voc(dump_data)
                # index += 1

            if conf['base']['gen_lsvt']:
                gen_lsvt(dump_data)
                # index += 1            
                # 
            if conf['base']['gen_icdar']:
                gen_icdar(dump_data)

            if conf['base']['gen_coco']:
                gen_coco(dump_data)
                # index += 1
            index += 1

            IMAGE_INDEX += 1


def gen_pic():
    from service import layout_provider
    layout = layout_provider.gen_next_layout()

    if not layout.is_empty():
        dump_data = layout.dump()
        layout.show(draw_rect=True)
        return dump_data
    else:
        log.info("-" * 10 + "layout is empty" + "-" * 10)
        return None


def add_label_data(layout_data):
    """
    Write label file
    :return:
    """
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    label_data_dir = get_label_data_dir(out_put_dir=out_put_dir)
    os.makedirs(label_data_dir, exist_ok=True)

    label_file_path = os.path.join(label_data_dir, "label_{pid}.txt".format(pid=os.getpid()))
    fragment_dir = get_fragment_dir(out_put_dir)

    fragment_list = layout_data['fragment']
    with open(label_file_path, 'a+') as f:
        for fragment in fragment_list:
            fragment_name = fragment['fragment_name']
            fragment_img_src_path = os.path.join(fragment_dir, fragment_name)
            fragment_img_dst_path = os.path.join(label_data_dir, fragment_name)
            shutil.copy(fragment_img_src_path, fragment_img_dst_path)

            txt = fragment['data']
            img_name = fragment['fragment_name']
            line = img_name + "^" + txt + os.linesep
            f.write(line)
    log.info("gen label data success!")


def gen_coco(layout_data):
    """
    Generate COCO data set
    :return:
    """
    global IMAGE_INDEX

    print(f'INDEX :: {IMAGE_INDEX}')
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    data_dir = get_coco_data_dir(out_put_dir=out_put_dir)
    data_dir_images = os.path.join(data_dir, "images")

    os.makedirs(data_dir_images, exist_ok=True)

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(data_dir_images, pic_name)

    # Copy picture
    shutil.copy(pic_path, pic_save_to_path)
    log.info("copy img success")
    # Generate label text
    json_path = os.path.join(data_dir, "train_{pid}.json".format(pid=os.getpid()))
    json_path = os.path.join(data_dir, "train.json")

    _gen_coco(layout_data, json_path)
    log.info("coco data gen success")


def _gen_coco(layout_data, json_path):
    """
    :param layout_data:
    :param json_path:
    :return:
    """
    global IMAGE_INDEX
    global ANNOTATION_INDEX
    print(f'IMAGE_INDEX / ANNOTATION_INDEX : {IMAGE_INDEX}  : {ANNOTATION_INDEX}')

    # print(layout_data)
    pic_name_full = layout_data['pic_name']
    width = layout_data['width']
    height = layout_data['height']
    pic_name = pic_name_full.split('.')[0]
    fragment_list = layout_data['fragment']

    if not os.path.exists(json_path):
        fp = open(json_path, "w")
        fp.close()

    with open(json_path, 'r') as f:
        text = f.read()
        if text == '':
            load_dict = dict()
            load_dict["categories"] = list()
            load_dict["images"] = list()
            load_dict["annotations"] = list()
            load_dict["categories"].append(
                {
                    "id": 1,
                    "name": "text",
                    "supercategory": "text"
                }
            )
        else:
            load_dict = json.loads(text)

    img_info = {
        "id": pic_name,
        "width": width,
        "height": height,
        "file_name": f'{pic_name_full}',
    }

    load_dict["images"].append(img_info)


    with open(json_path, 'w') as f:
        annon_dict_list = load_dict["annotations"]

        for fragment in fragment_list:
            # print(fragment)
            txt = fragment['data']
            fragment_name = fragment['fragment_name']
            rotate_box = fragment['rotate_box']
            char_boxes = fragment['char_boxes']

            hash_object = hashlib.sha256(str(fragment_name).encode('utf-8'))
            fid = hash_object.hexdigest()

            rotate_rect_tuple = []
            rotate_rect_tupleXX = []
            for point in rotate_box:
                rotate_rect_tuple.append(point[0])
                rotate_rect_tuple.append(point[1])

            contour = np.array(char_boxes, dtype=np.int)
            min_x = np.amin(contour[:, :, 0])
            max_x = np.amax(contour[:, :, 0])
            min_y = np.amin(contour[:, :, 1])
            max_y = np.amax(contour[:, :, 1])

            w = max_x - min_x
            h = max_y - min_y
            _x0, _y0 = min_x, min_y
            _x1, _y1 = min_x, min_y + h
            _x2, _y2 = min_x + w, min_y + h
            _x3, _y3 = min_x + w, min_y

            # print('****')
            print("{},{},{},{} : {}, {}".format(min_x, max_x, min_y, max_y, w, h))

            img_info = {
                "id": fid,
                "image_id": pic_name,
                "category_id": 1,
                # "segmentation": [[490.29, 333.78, 609.52, 333.54, 609.76, 357.73, 489.34, 356.77]],
                # "bbox": [489.34, 333.54, 120.42, 24.19]
                "segmentation_boxed": [[int(_x0), int(_y0), int(_x1), int(_y1), int(_x2), int(_y2), int(_x3), int(_y3)]],
                "segmentation": [rotate_rect_tuple],
                "bbox": [int(min_x), int(min_y), int(w), int(h)]
            }

            annon_dict_list.append(img_info)
            ANNOTATION_INDEX += 1

        json.dump(load_dict, f, indent=4)


def gen_voc(layout_data):
    """
    Generate voc data set
    :return:
    """
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    voc_data_dir = get_voc_data_dir(out_put_dir=out_put_dir)

    voc_img_dir = os.path.join(voc_data_dir, "voc_img")
    voc_xml_dir = os.path.join(voc_data_dir, "voc_xml")
    os.makedirs(voc_img_dir, exist_ok=True)
    os.makedirs(voc_xml_dir, exist_ok=True)

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(voc_img_dir, pic_name)

    # Copy picture
    shutil.copy(pic_path, pic_save_to_path)
    log.info("copy img success")

    # Generate label text
    _gen_voc(voc_xml_dir, data=layout_data)

    log.info("voc data gen success")


def _gen_voc(save_dir, data, image_format='png'):
    w = data['width']
    h = data['height']

    node_root = Element('annotation')
    '''folder'''
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    '''filename'''
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = data['pic_name']
    '''source'''
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'The VOC2007 Database'
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = 'PASCAL VOC2007'
    node_image = SubElement(node_source, 'image')
    node_image.text = 'flickr'
    '''size'''
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    '''segmented'''
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    '''object coord and label'''
    for i, fragment in enumerate(data['fragment']):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = fragment['orientation'][0] + "_text"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(fragment['box'][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(fragment['box'][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(fragment['box'][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(fragment['box'][3])

    xml = tostring(node_root, pretty_print=True)  # Format display, the newline of the newline

    save_xml = os.path.join(save_dir, data['pic_name'].replace(image_format, 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)


def imwrite(path, img):
    try:
        print(path)
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)


def gen_icdar(layout_data):
    """
    Generate ICDAR format
    :param layout_data:
    :return:
    """

    # mask_img = Image.new('RGBA', self.bg_img.size, (255, 255, 255, 0))
    # name = hashlib.sha1(mask_img.tobytes()).hexdigest()
    # pic_name = "pic_" + name + ".png"
    # pic_dir ='/tmp/pics2'

    # # convert from RGBA->RGB 
    # background = Image.new('RGB', mask_img.size, (255,255,255))
    # background.paste(mask_img, mask = mask_img.split()[3])
    # inv_img = ImageOps.invert(background)

    # pic_path = os.path.join(pic_dir, pic_name)
    # with open(pic_path, 'wb') as f:
    #     inv_img.save(f, "png")

    print("Generating ICDAR dataformat")
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    icdr_data_dir = get_icidar_data_dir(out_put_dir=out_put_dir)
    icdar_data_img_dir = os.path.join(icdr_data_dir)
    os.makedirs(icdar_data_img_dir, exist_ok=True)

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(icdar_data_img_dir, pic_name)
    # Copy picture
    # shutil.copy(pic_path, pic_save_to_path)

    im = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    bin = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)[1]
    imwrite(pic_save_to_path, bin)
    log.info("copy img success")
    # Generate label text
    # _gen_icdar(layout_data)
    name = pic_name.split('.')[0]
    icidar_label_path = os.path.join(icdr_data_dir, "gt_{name}.txt".format(name=name))
    log.info("ICIDAR data gen success")
    # _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, txt
    fragment_list = layout_data['fragment']

    with open(icidar_label_path, 'w') as f:
        for fragment in fragment_list:
            txt = fragment['data']
            rotate_box = fragment['rotate_box']
            char_boxes = fragment['char_boxes']
            contour = np.array(char_boxes, dtype=np.int)

            # print('--' * 80)
            # print (rotate_box)
            # # print (char_boxes)
            # print('--' * 25)
            # for box in char_boxes:
            #     print(box)

            min_x = np.amin(contour[:, :, 0])
            max_x = np.amax(contour[:, :, 0])
            min_y = np.amin(contour[:, :, 1])
            max_y = np.amax(contour[:, :, 1])

            w = max_x - min_x
            h = max_y - min_y
            _x0, _y0 = min_x, min_y
            _x1, _y1 = min_x, min_y + h
            _x2, _y2 = min_x + w, min_y + h
            _x3, _y3 = min_x + w, min_y

            print("{},{},{},{},{},{},{},{},{}".format(_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt))
            f.write("{},{},{},{},{},{},{},{},{}\n".format(_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt))
            continue
            # vmin = np.amin(char_boxes, axis=1)
            print('****')
            print("{},{},{},{} : {}, {}".format(min_x, max_x, min_y, max_y, w, h))
            # mar = cv2.minAreaRect(contour)

            os.system.exit()
            # print (char_boxes)
            # print("txt = {txt} : {box}".format(txt=txt, box=box))
            _x0, _y0 = rotate_box[0][0], rotate_box[0][1]
            _x1, _y1 = rotate_box[1][0], rotate_box[1][1]
            _x2, _y2 = rotate_box[2][0], rotate_box[2][1]
            _x3, _y3 = rotate_box[3][0], rotate_box[3][1]
            f.write("{},{},{},{},{},{},{},{},{}\n".format(_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt))


def gen_lsvt(layout_data):
    """

    :param layout_data:
    :return:
    """
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    lsvt_data_dir = get_lsvt_data_dir(out_put_dir=out_put_dir)

    lsvt_data_img_dir = os.path.join(lsvt_data_dir, "train")
    os.makedirs(lsvt_data_img_dir, exist_ok=True)
    lsvt_json_path = os.path.join(lsvt_data_dir, "train_full_labels_{pid}.json".format(pid=os.getpid()))

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(lsvt_data_img_dir, pic_name)
    # Copy picture
    shutil.copy(pic_path, pic_save_to_path)
    log.info("copy img success")
    # Generate label text
    _gen_lsvt(layout_data, lsvt_json_path)
    log.info("voc data gen success")


def _gen_lsvt(layout_data, lsvt_json_path):
    """

    :param layout_data:
    :param lsvt_json_path:
    :return:
    """
    pic_name = layout_data['pic_name']
    pic_name = pic_name.split('.')[0]
    fragment_list = layout_data['fragment']
    print(lsvt_json_path)
    if not os.path.exists(lsvt_json_path):
        fp = open(lsvt_json_path, "w")
        fp.close()
    with open(lsvt_json_path, 'r') as f:
        text = f.read()
        if text == '':
            load_dict = dict()
        else:
            load_dict = json.loads(text)

    with open(lsvt_json_path, 'w') as f:
        lsvt_dict_list = list()
        for fragment in fragment_list:
            txt = fragment['data']
            rotate_box = fragment['rotate_box']
            char_boxes = fragment['char_boxes']
            lsvt_info = dict(transcription=txt, points=rotate_box, char_boxes=char_boxes, illegibility=False)
            lsvt_dict_list.append(lsvt_info)
        load_dict.update({pic_name: lsvt_dict_list})
        # f.seek(0)

        json.dump(load_dict, f)
