import torch
from model import *

import torchvision.transforms as transforms
from torch import nn
import os
import numpy as np
from PIL import Image
import cv2
from CAM import GradCAM
from run import sx_model_index, mx_model_index, size

label = {0: "Feiyinxing", 1: "Yinxing"}
# 数据转换器
data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
root_dir = r"."
batch_size = 1
weight_dir = os.path.join(root_dir, "weight")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xuxu = MultResNet(sx_model_index, mx_model_index).to(device)
model_name = "MultResNet"

if weight_dir is not None and model_name + ".pth" in os.listdir(weight_dir):
    try:
        xuxu.load_state_dict(torch.load(os.path.join(weight_dir, model_name + ".pth")))
    except:
        print(
            "****(= 7 ^ 7 =)---- {}: model structure has been changed!!! ----(T^T)----".format(
                model_name
            )
        )
    else:
        print(
            "\/\/\/(= ^ _ ^ =)//// {}: model load successfully !!!".format(model_name)
        )

sx_layers = [
    xuxu.features1[0],
    xuxu.features1[4][0].conv1,
    xuxu.features1[4][1].conv3,
    xuxu.features1[4][2].conv3,
    xuxu.features1[5][0].conv3,
    xuxu.features1[5][1].conv3,
    xuxu.features1[5][2].conv3,
    xuxu.features1[5][3].conv3,
    xuxu.features1[6][0].conv3,
    xuxu.features1[6][1].conv3,
    xuxu.features1[6][2].conv3,
    xuxu.classifier1[0][0].conv3,
    xuxu.classifier1[0][1].conv3,
    xuxu.classifier1[0][2].conv3,
]
mx_layers = [
    xuxu.features2[0],
    xuxu.features2[4][0].conv1,
    xuxu.features2[4][1].conv3,
    xuxu.features2[4][2].conv3,
    xuxu.features2[5][0].conv3,
    xuxu.features2[5][1].conv3,
    xuxu.features2[5][2].conv3,
    xuxu.features2[5][3].conv3,
    xuxu.features2[6][0].conv3,
    xuxu.features2[6][1].conv3,
    xuxu.features2[6][2].conv3,
    xuxu.classifier2[0][0].conv3,
    xuxu.classifier2[0][1].conv3,
    xuxu.classifier2[0][2].conv3,
]
target_layers = [sx_layers, mx_layers]
gradcam = GradCAM(xuxu, target_layers)


def overlap(cam, image, size):
    image = np.array(image)
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    cam = nn.functional.interpolate(
        cam, size=size, mode="bilinear", align_corners=False
    )
    cam = cam.cpu().detach().numpy()[0, 0]
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, cam, 0.5, 0)
    # result = cv2.resize(result, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    return image, result


def deal(images, size=(128, 128)):
    images = [image.resize(size, Image.ANTIALIAS) for image in images]
    x = [data_transform(np.array(image)) for image in images]
    x = [i.unsqueeze(0).to(device) for i in x]
    original_cam, class_idx = gradcam(x) #得到舌象的类激活图和大张目象的类激活图
    sxcams, mxcams = [original_cam[0]], []
    for mxcam in original_cam[1]:
        cams = []
        h, w = mxcam.shape[-2:]  # 切割类激活图
        for i in range(0, h, h // 2):
            for j in range(0, w, w // 5):
                cams.append(mxcam[:, :, i : i + h // 2, j : j + w // 5])
        mxcams.append(cams)
    tt = [[] for i in range(10)] #翻转类激活图
    for i in range(len(mxcams)):
        for j in range(len(mxcams[i])):
            tt[j].append(mxcams[i][j])
    mxcams = tt
    cams = sxcams + mxcams
    for i in range(len(cams)):
        for j in range(len(cams[i])):
            images[i], cams[i][j] = overlap(cams[i][j], images[i], size=size)

    return cams, images, label[class_idx]


if __name__ == "__main__":
    patient_path = r"patient/Feiyinxing/303"

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # 准备数据
    images = [Image.open(os.path.join(patient_path + ".jpg"))] + [
        Image.open(os.path.join(patient_path, i)) for i in os.listdir(patient_path)
    ]
    # images = [image.resize((128, 128), Image.ANTIALIAS) for image in images]
    # x = [data_transform(np.array(image)) for image in images]

    # x = [i.unsqueeze(0).to(device) for i in x]

    # cams, class_idx = gradcam(x)
    results, images, label = deal(images, size=(64, 64))
    # print(label[class_idx])
    # 将 CAM 图像与原始图像叠加

    # for i in range(len(cams)):
    #     images[i], cams[i] = overlap(cams[i], images[i])
    # cv2.imshow("original",cv2.resize(np.array(images[1]), dsize=(512, 512), interpolation=cv2.INTER_CUBIC))
    for i in range(len(results)):
        cv2.imshow("images" + str(i), images[i])
        cv2.imshow("result" + str(i), results[i])
    cv2.waitKey(0)
