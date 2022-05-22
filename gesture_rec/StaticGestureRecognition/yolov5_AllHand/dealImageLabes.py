import os
from shutil import copy

labels_path = 'runs/detect/exp'
images_path = 'E:/DataSet/Mask/'

new_labels_path = 'D:/Python/ML/YOLO/Mask_Face/3classes/labels/'
new_images_path = 'D:/Python/ML/YOLO/Mask_Face/3classes/images/'

mul_labels_path = 'D:/Python/ML/YOLO/Mask_Face/3class/labels/'
mul_images_path = 'D:/Python/ML/YOLO/Mask_Face/3class/images/'

# 将图像和标签复制一份
# for i in range(10, 20):
#     sin_labels_path = labels_path + str(i)+'/labels/'
#     sin_images_path = images_path + str(i) + '000/'
#
#     labels_file = os.listdir(sin_labels_path)
#     images_file = os.listdir(sin_images_path)
#
#     for label_file in labels_file:
#         full_label_path = sin_labels_path+label_file
#         print(full_label_path)
#         copy(full_label_path, new_labels_path)
#
#     for image_file in images_file:
#         full_image_path = sin_images_path+image_file
#         print(full_image_path)
#         copy(full_image_path, new_images_path)


# 交叉验证是否存在无标签或无图像文件
# labels = os.listdir(new_labels_path)
# images = os.listdir(new_images_path)
#
# for image in images:
#     txt_name = image[:-3] + 'txt'
#     if txt_name not in labels:
#         print(image)
#
# for label in labels:
#     image_name = label[:-3] + 'jpg'
#     if image_name not in images:
#         print(label)

# 将标签0改为1
# labels = os.listdir(new_labels_path)
# for label in labels:
#     full_label_path = new_labels_path + label
#     new_text = []
#     with open(full_label_path) as f1:
#         context = f1.readlines()
#         for line_text in context:
#             new_text.append('1' + line_text[1:])
#     with open(full_label_path, 'w+') as f2:
#         for text in new_text:
#             f2.write(text)
#     print(f'{label}写入成功！')


# 批量重命名
# labels = os.listdir(new_labels_path)
# images = os.listdir(new_images_path)
#
# label_index = 21221
# image_index = 21221
#
# for label in labels:
#     old_name = new_labels_path + label
#     new_name = new_labels_path + str.zfill(str(label_index), 5) + '.txt'
#     # print(f'oldname: {old_name}')
#     # print(f'newname: {new_name}')
#     label_index += 1
#     os.rename(old_name, new_name)
#     print(f'{new_name}命名完成！')
#
# for image in images:
#     old_name = new_images_path + image
#     new_name = new_images_path + str.zfill(str(image_index), 5) + '.jpg'
#     # print(f'oldname: {old_name}')
#     # print(f'newname: {new_name}')
#     image_index += 1
#     os.rename(old_name, new_name)
#     print(f'{new_name}命名完成！')


# 查看是否有标签为0
# labels = os.listdir(mul_labels_path)
# images = os.listdir(mul_images_path)
#
# for label in labels:
#     full_label_path = mul_labels_path + label
#     with open(full_label_path) as f:
#         context = f.readlines()
#         for text in context:
#             text_list = text.split(' ')
#             if text_list[3] == '0.0' or text_list[4][:-1] == '0.0':
#                 print(full_label_path)

# 将未上传过的图提取出来
num = 13109
labels = os.listdir(mul_labels_path)
images = os.listdir(mul_images_path)
count = 0
for image in images:
    count += 1
    if count >= num:
        full_path = mul_images_path + image
        print(full_path)
        # print(mul_images_path[:-7]+'backup/')
        copy(full_path, mul_images_path[:-7] + 'backup/')
