'''
将coco格式的runway数据集中的annotations下的json文件中的四个角点，处理成yolo格式的x,y,w,h包围框 （非归一化）
然后重新存储成json

使用方法：
指定原json及处理过之后要保存的json文件即可

'''
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--origin_json_path', default='./annotations/train.json',type=str, help='the path of the anns json')
parser.add_argument('--new_json_path', default='./annotations/new_train.json', type=str, help='the path of the saved json path')
arg = parser.parse_args()

#read json file
with open(arg.origin_json_path, "r") as f:
    json_labels = json.load(f)

# points to bbox
# train.json
for anno in tqdm(json_labels["annotations"]):
    bbox_list = anno["bbox"]
    max_x, min_x, max_y, min_y = 0, 9e4, 0, 9e4
    for point in bbox_list:
        max_x = max(max_x, point[0])
        max_y = max(max_y, point[1])
        min_x = min(min_x, point[0])
        min_y = min(min_y, point[1])
    anno["bbox"] = [min_x, min_y, max_x - min_x, max_y - min_y]


#wirte json
with open(arg.new_json_path, "w") as f:
    json.dump(json_labels, f)


print("transfer is complete!")