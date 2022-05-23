import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
sys.path.append(current_directory)

from PIL import Image, ImageFile, ImageFilter, ImageEnhance, ImageOps
from eye_track.remove_glass.misc import get_potrait, upscale
import io


import tflite_runtime.interpreter as tflite

import torch
import cv2
import contextlib
from eye_track.remove_glass.data.base_dataset import get_transform
from eye_track.remove_glass.models.cut_model import CUTModel
from eye_track.remove_glass.util.util import tensor2im

from argparse import Namespace
from pathlib import Path
from copy import deepcopy
from Models import *
import matplotlib.pyplot as plt

# Initialize the tflie interpreter for potrait segmentation
interpreter = tflite.Interpreter(
    model_path=current_directory + "/model_check_points/slim_reshape_v2.tflite"
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

OPT = Namespace(
    batch_size=1,
    checkpoints_dir="cyclegan",
    crop_size=256,
    # dataroot=".",
    dataset_mode="unaligned",
    direction="AtoB",
    display_id=-1,
    display_winsize=256,
    epoch="latest",
    eval=False,
    gpu_ids=[],
    nce_layers="0,4,8,12,16",
    nce_idt=False,
    lambda_NCE=10.0,
    lambda_GAN=1.0,
    init_gain=0.02,
    nce_includes_all_negatives_from_minibatch=False,
    init_type="xavier",
    normG="instance",
    no_antialias=False,
    no_antialias_up=False,
    netF="mlp_sample",
    netF_nc=256,
    nce_T=0.07,
    num_patches=256,
    CUT_mode="FastCUT",
    input_nc=3,
    isTrain=False,
    load_iter=0,
    load_size=256,
    max_dataset_size=float("inf"),
    model="CUT",
    n_layers_D=3,
    name=None,
    ndf=64,
    netD="basic",
    netG="resnet_9blocks",
    ngf=64,
    no_dropout=True,
    no_flip=True,
    num_test=50,
    num_threads=4,
    output_nc=3,
    phase="test",
    preprocess="scale_width",
    random_scale_max=3.0,
    results_dir="./results/",
    serial_batches=True,
    suffix="",
    verbose=False,
)

model_cran_v2 = CARN_V2(
    color_channels=3,
    mid_channels=64,
    conv=nn.Conv2d,
    single_conv_size=3,
    single_conv_group=1,
    scale=2,
    activation=nn.LeakyReLU(0.1),
    SEBlock=True,
    repeat_blocks=3,
    atrous=(1, 1, 1),
)

model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = current_directory + "/model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, "cpu"))
# if use GPU, then comment out the next line so it can use fp16.
model_cran_v2 = model_cran_v2.float()

fp = current_directory + "/cyclegan/EyeFastcut/latest_net_G.pth"
opt = deepcopy(OPT)
model_name = "EyeFastcut"
opt.name = model_name
if opt.verbose:
    # model = load_model(opt, model_fp)
    model = CUTModel(opt).netG
    model.load_state_dict(torch.load(fp))
else:
    with contextlib.redirect_stdout(io.StringIO()):
        # model = load_model(opt, model_fp)
        model = CUTModel(opt).netG
        model.load_state_dict(torch.load(fp))


class SingleImageDataset(torch.utils.data.Dataset):
    """dataset with precisely one image"""

    def __init__(self, img, preprocess):
        img = preprocess(img)
        self.img = img

    def __getitem__(self, i):
        return self.img

    def __len__(self):
        return 1


def cutgan(img: Image) -> Image:
    img = img.convert("RGB")
    data_loader = torch.utils.data.DataLoader(
        SingleImageDataset(img, get_transform(opt)), batch_size=1
    )
    data = next(iter(data_loader))
    with torch.no_grad():
        pred = model(data)
    pred_arr = tensor2im(pred)
    pred_img = Image.fromarray(pred_arr)
    return pred_img


def remove_glass(im):
    # im = ImageOps.exif_transpose(im)
    im = Image.fromarray(im)
    width, height = im.size
    ori_im = im.copy()

    # get potrait and mask
    im, mask = get_potrait(im, interpreter, input_details, output_details)

    # send image to model to remove glasses
    im = cutgan(im)
    # plt.imshow(im)
    # plt.show()
    # composite original image and output based on mask
    w, h = im.size
    ori_im = ori_im.resize((w, h))
    mask = mask.resize((w, h))
    img = Image.composite(im, ori_im, mask)

    # upscale the image
    img = upscale(img, model_cran_v2, path=current_directory + "/asset/removal.png")
    img = img.resize((width, height))
    # scale image to original size
    img = np.array(img)
    # cv2.imshow("111", img)
    # cv2.waitKey(10)
    return img

def forimg():
    if __name__ == '__main__':
        im = Image.open(current_directory + "/2.jpg")
        # im = im.resize((1280, 960))
        im = ImageOps.exif_transpose(im)
        width, height = im.size
        ori_im = im.copy()

        # get potrait and mask
        im, mask = get_potrait(im, interpreter, input_details, output_details)

        # send image to model to remove glasses
        im = cutgan(im)
        plt.imshow(im)
        plt.show()
        # composite original image and output based on mask
        w, h = im.size
        ori_im = ori_im.resize((w, h))
        mask = mask.resize((w, h))
        img = Image.composite(im, ori_im, mask)
        # upscale the image
        img = upscale(img, model_cran_v2, path=current_directory+"/removal.png")
        print(img.size)
        # scale image to original size
        img = img.resize((width, height))
        img.save(current_directory+"/removal.png")

def forVideo():


    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
    while True:
        _, img = cap.read()
        cv2.imshow("1", img)
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # im = Image.open(current_directory + "/asset/my.jpg")

        im = ImageOps.exif_transpose(im)
        width, height = im.size
        print(width, height)
        ori_im = im.copy()

        # get potrait and mask
        im, mask = get_potrait(im, interpreter, input_details, output_details)

        # send image to model to remove glasses
        im = cutgan(im)
        # plt.imshow(im)
        # plt.show()
        # composite original image and output based on mask
        w, h = im.size
        ori_im = ori_im.resize((w, h))
        mask = mask.resize((w, h))
        img = Image.composite(im, ori_im, mask)

        # upscale the image
        img = upscale(img, model_cran_v2, path=current_directory + "/asset/removal.png")
        print(img.size)
            # scale image to original size
        img = img.resize( (width, height))
        print(img)
        # img = np.transpose(img, (1,2,0))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("111", img)
        # cv2.moveWindow("111",10,10)
        cv2.waitKey(10)

if __name__ == '__main__':
    forVideo()
    # forimg()



    # img.save(current_directory + "/asset/removal2.png")