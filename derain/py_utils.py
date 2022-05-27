import os, cv2, shutil
from PyQt5.QtGui import QPixmap, QImage
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

# 删除文件夹下所有文件
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
def sync_video(file_dir, video_name='output.avi'):
    filelist = read_from_file_k(file_dir)
    size = (256, 256)
    for item in filelist:
        if item.endswith('.png'):
            shape = cv2.imread(item).shape
            size = (shape[1], shape[0])
            break
    # print(size)
    fps = 10  # 我设定位视频每秒1帧，可以自行修改
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data/output/"
    video = cv2.VideoWriter(path + video_name, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.png'):
            # print(item)
            img = cv2.imread(item)
            img = cv2.resize(img, size)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('视频合成生成完成')

# 列出当前目录下的所有文件和目录名
def list_all_filename():
    return [d for d in os.listdir('.')]

# 输出某个路径下的所有文件和文件夹的路径
def list_all_dir(filepath):
    list = []
    for i in os.listdir(filepath):  # 获取目录中文件及子目录列表
        #print(os.path.join(filepath, i))   # 把路径组合起来
        list.append(os.path.join(filepath, i))
    return list

# 输出某个路径及其子目录下的所有文件路径
def list_all_seed_dir(filepath):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        print(path)
        if os.path.isdir(path):
            list_all_seed_dir(path)      # 递归

# 输出某个路径及其子目录下所有以.mp4为后缀的文件
def print_dir (filepath, end=".mp4"):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            print_dir(path)
        if path.endswith(end):
            print(path)


#写文件
def write_totxt(filepath="test.txt"):
    with open (filepath, "wt") as outfile:
        outfile.write("该文本会写入到文件中 n 看到我了吧！")

#Read a file
def read_file(filepath="test.txt"):
    with open(filepath, "rt") as infile:
        return infile.read()









