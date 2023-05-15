'''
对coco数据集进行可视化
使用方法： 填写json文件所在地址   以及 对应文件夹即可完成可视化

'''

import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

json_path = "../COCO/annotations/instances_val2017.json"
img_path = "../COCO/val2017"
with open(json_path,'r') as f:
    json_labels = json.load(f)


# load downloadFromYolov6 data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# list  = [(v["id"], v["name"]) for k, v in downloadFromYolov6.cats.items()]
# get all downloadFromYolov6 class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

# 遍历前三张图像
for img_id in ids[:3]:
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)

    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)

    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']

    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    # draw box to image
    for target in targets:
        x, y, w, h = target["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])

    # show image
    plt.imshow(img)
    plt.show()
