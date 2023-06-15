"""
coco格式转换为yolo格式,将runway的anno转化成yolo格式的label,格式为：cls、xywh、6 points

author: Wu
https://github.com/Weifeng-Chen/DL_tools/issues/3

COCO 格式的数据集转化为 YOLO 格式的数据集，源代码采取遍历方式，太慢，
这里改进了一下时间复杂度，从O(nm)改为O(n+m)，但是牺牲了一些内存占用
--json_path 输入的json文件路径
--save_path 保存的文件夹名字，默认为当前目录下的labels。


使用方法：
指定annotations文件夹和你需要保存的labels文件夹即可，注意coco数据集一般有一个train.json和val.json，所以我们需要执行两遍这个代码

coco格式的数据集中标注物体的格式是xyxy 原图尺度

"""

import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='/workspace/YOLOv6/data/runway_filter/annotations/val.json', type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='/workspace/YOLOv6/data/runway_filter/labels/val', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

# 将box的x_lt y_lt w h（原图尺度）  --->  x_c y_c w h（归一化）
def box_convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def point_convert(size, point_list):
    new_point_list =[]
    for point in point_list:
        new_point_list.append([point[0] * 1.0 / size[0], point[1] * 1.0 / size[1]])
    
    return new_point_list

#根据4个点的坐标，得到最小包围框的x_lt y_lt w h （原图尺度）
def get_xywh_from_points(points):
    max_x, max_y, min_x, min_y = -1, -1, 2e5, 2e5
    for point in points:
        max_x = max(max_x, point[0])
        min_x = min(min_x, point[0])
        max_y = max(max_y, point[1])
        min_y = min(min_y, point[1])
    
    return min_x, min_y, max_x- min_x, max_y - min_y 


if __name__ == '__main__':
    json_file = arg.json_path  # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    for i, category in enumerate(data['categories']):
        id_map[category['id']] = i

    # 通过事先建表来降低时间复杂度
    max_id = 0
    for img in data['images']:
        max_id = max(max_id, img['id'])
    # 注意这里不能写作 [[]]*(max_id+1)，否则列表内的空列表共享地址
    img_ann_dict = [[] for i in range(max_id + 1)]
    for i, ann in enumerate(data['annotations']):
        img_ann_dict[ann['image_id']].append(i) # img_ann_dict是一个双层list，记录的是图片id对应的anno id 。 由于runway数据特殊，这里应该是1对1

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        # img_id = 6176
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        '''for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))'''
        # 这里可以直接查表而无需重复遍历
        
        for ann_id in img_ann_dict[img_id]:
            ann = data['annotations'][ann_id]
            points_4 = ann["bbox"]
            center_line_points = ann["center_line"]
            xywh = get_xywh_from_points(points_4)
            box = box_convert((img_width, img_height), xywh)# x_lt y_lt w h（原图尺度）  --->  x_c y_c w h（归一化）
            points_4 = point_convert((img_width, img_height), points_4)
            center_line_points = point_convert((img_width, img_height), center_line_points)

            # 因此最终的label文件的box的四个值是x_c y_c w h（归一化），6个点是归一化之后的
            f_txt.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % \
                        (id_map[ann["category_id"]], box[0], box[1], box[2], box[3],\
                         points_4[0][0],points_4[0][1],points_4[1][0],points_4[1][1],points_4[2][0],points_4[2][1],points_4[3][0],points_4[3][1],\
                         center_line_points[0][0],center_line_points[0][1],center_line_points[1][0],center_line_points[1][1]\
                            ))
        f_txt.close()

    print(f"transfer is complete, labels saved at {ana_txt_save_path}")

'''

# 旧版，很慢hhh
# """
# COCO 格式的数据集转化为 YOLO 格式的数据集
# --json_path 输入的json文件路径
# --save_path 保存的文件夹名字，默认为当前目录下的labels。
# """

# import os
# import json
# from tqdm import tqdm
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--json_path', default='./instances_val2017.json',type=str, help="input: coco format(json)")
# parser.add_argument('--save_path', default='./labels', type=str, help="specify where to save the output dir of labels")
# arg = parser.parse_args()

# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = box[0] + box[2] / 2.0
#     y = box[1] + box[3] / 2.0
#     w = box[2]
#     h = box[3]

#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)

# if __name__ == '__main__':
#     json_file =   arg.json_path # COCO Object Instance 类型的标注
#     ana_txt_save_path = arg.save_path  # 保存的路径

#     data = json.load(open(json_file, 'r'))
#     if not os.path.exists(ana_txt_save_path):
#         os.makedirs(ana_txt_save_path)

#     id_map = {} # coco数据集的id不连续！重新映射一下再输出！
#     with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
#         # 写入classes.txt
#         for i, category in enumerate(data['categories']):
#             f.write(f"{category['name']}\n")
#             id_map[category['id']] = i
#     # print(id_map)

#     for img in tqdm(data['images']):
#         filename = img["file_name"]
#         img_width = img["width"]
#         img_height = img["height"]
#         img_id = img["id"]
#         head, tail = os.path.splitext(filename)
#         ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
#         f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
#         for ann in data['annotations']:
#             if ann['image_id'] == img_id:
#                 box = convert((img_width, img_height), ann["bbox"])
#                 f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
#         f_txt.close()
'''
