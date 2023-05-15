'''
yolo数据可视化

使用方法：需要指定图片文件夹（下一级目录就是图片）和labels文件夹（下一层就是txt）以及标好bbox框的输出文件夹

'''

import os
import cv2
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--labels_path", default="./labels/val", help="the path of label dir which subplace is the txt")
parser.add_argument("--imgs_path", default="./images/val", help="the path of image dir which subplace is the jpg")
parser.add_argument("--output_path", default="./output", help="the path of output jpg dir ")
arg = parser.parse_args()

def main():
	# 总的检测根目录
    path_root_labels = arg.labels_path
    # 总的检测根目录
    path_root_imgs = arg.imgs_path
    path_sava_imgs = arg.output_path
    if not os.path.exists(path_sava_imgs):
        os.makedirs(path_sava_imgs)
    type_object = '.txt'


    for ii in os.walk(path_root_imgs):
        for j in ii[2]: #
            type = j.split(".")[1]
            if type != 'jpg':
                continue
            path_img = os.path.join(path_root_imgs, j)
            # print(path_img)
            label_name = j[:-4]+type_object
            path_label = os.path.join(path_root_labels, label_name)
            # print(path_label)
            f = open(path_label, 'r+', encoding='utf-8')
            if os.path.exists(path_label) == True:

                img = cv2.imread(path_img)
                w = img.shape[1]
                h = img.shape[0]
                new_lines = []
                img_tmp = img.copy()
                while True:
                    line = f.readline()
                    if line:
                        msg = line.split(" ")
                        # print(x_center,",",y_center,",",width,",",height)
                        x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                        y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                        x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                        y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                        # print(x1,",",y1,",",x2,",",y2)
                        cv2.rectangle(img_tmp,(x1,y1),(x2,y2),(0,0,255),5)
                    else :
                        break
            # cv2.imshow("show", img_tmp)
            # print(f"{path_sava_imgs}/{j}")
            cv2.imwrite(f"{path_sava_imgs}/{j}",img_tmp)
            # c = cv2.waitKey(0)



if __name__ == '__main__':
    startTime = time.time()
    print("draw start")
    main()
    endTime = time.time()
    print(f"draw end, it costs {round(endTime - startTime, 1)} seconds")


