import os, cv2, time, shutil
from PyQt5.QtGui import QPixmap, QImage
from tqdm import tqdm
from PIL import Image
import numpy as np

def read_pixmap(filename="nil", show_image=''):
    if filename == "nil":
        # 从本地读图
        return QPixmap(filename)
    else:
        # np数组生成的图
        len_x = show_image.shape[1]  # 获取图像大小
        wid_y = show_image.shape[0]
        frame = QImage(show_image.data, len_x, wid_y, len_x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        return pix

def trainsform(video_path='/data/test.flv', out_path='data/test2.avi'):  # 自定义输出后缀
    cap = cv2.VideoCapture(video_path)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (weight, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, size)  # fourcc是编码格式，size是图片尺寸
    for _ in tqdm(range(frame_cnt)):
        #print('视频合成进度:', n, frame_cnt)
        ret, frame = cap.read()
        # cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()

def process_image(image):
    return image

# 读取视频并实时处理输出处理后的视频
def trainsfer_immediately():
    ######################     视频载入       #############################
    cap = cv2.VideoCapture("test.flv")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    size = (weight, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output11.avi', fourcc, 20, size)

    #####################       模型载入      #############################

    #####################      视频处理       #############################
    num = 0
    while cap.isOpened():
        # get a frame
        rval, frame = cap.read()
        # save a frame
        if rval == True:
            #  frame = cv2.flip(frame,0)
            # Start time
            start = time.time()
            #rclasses, rscores, rbboxes = process_image(frame) #换成自己调用的函数

            clean_image = process_image(frame)  # 换成自己调用的函数
            # End time

            end = time.time()
            # Time elapsed
            seconds = end - start + 0.0001
            print("Time taken : {0} seconds".format(seconds))
            # Calculate frames per second
            fps = 1 / seconds;
            print("Estimated frames per second : {0}".format(fps));
            # bboxes_draw_on_img(frame,rclasses,rscores,rbboxes)
            # print(rclasses)
            out.write(clean_image)
            num = num + 1
            print(num)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        else:
            break
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#保存CV2图片到本地
def save_to_file(filepath, filename, img):
    try:
        os.chdir(filepath)
    except Exception as e:
        print(e)
    cv2.imwrite(filename, img)


######################################################################################
##  Image utility
######################################################################################

def numpy_to_PIL(img_np):
    ## input image is numpy array in [0, 1]
    ## convert to PIL image in [0, 255]

    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)

    return img_PIL


def PIL_to_numpy(img_PIL):
    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0

    return img_np


def read_img(filename, grayscale=0):
    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = img[:, :, ::-1]  ## BGR to RGB

    img = np.float32(img) / 255.0

    return img


def save_img(img, filename):
    print("Save %s" % filename)

    if img.ndim == 3:
        img = img[:, :, ::-1]  ### RGB to BGR

    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# 读取目录下的非隐藏文件, 并对其排序
def read_from_file_k(file_dir):
    filelist = os.listdir(file_dir)
    filelist = [f for f in filelist if not file_is_hidden(f)]# 删除隐藏文件
    filelist.sort(key=lambda x:int(x.split('.')[0]))
    filelist2 = [os.path.join(file_dir, f) for f in filelist]
    return filelist2

def file_is_hidden(p):
    return p.startswith('.')

# 合成视频
def sync_video(file_dir, video_name='output.mp4'):
    filelist2 = read_from_file_k(file_dir)
    size = (256, 256)
    for item in filelist2:
        if item.endswith('.png'):
            shape = cv2.imread(item).shape
            size = (shape[1], shape[0])
            break
    # print(size)
    fps = 10  # 我设定位视频每秒1帧，可以自行修改
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("./data/output/" + video_name, fourcc, fps, size)

    for item in filelist2:
        if item.endswith('.png'):
            # print(item)
            img = cv2.imread(item)
            img = cv2.resize(img, size)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('视频合成生成完成')



if __name__ == "__main__":
    trainsform()