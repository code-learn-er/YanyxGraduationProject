import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

# 定义 GradCAM 方法的实现函数。

import numpy as np


class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.feature_maps, self.grads = [], {}

        self.forward_hook, self.backward_hook = [], []

        self.model.eval()
        for i in range(len(self.target_layers)):
            for j in range(len(self.target_layers[i])):
                self.forward_hook.append(
                    self.target_layers[i][j].register_forward_hook(
                        self.save_feature_maps
                    )
                )
                self.backward_hook.append(
                    self.target_layers[i][j].register_backward_hook(self.save_grads)
                )

    def save_feature_maps(self, module, input, output):
        self.feature_maps.append((id(module), output.detach()))

    def save_grads(self, module, grad_input, grad_output):
        self.grads[id(module)] = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.feature_maps = []
        self.grads = {}
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = np.argmax(output.cpu().detach().numpy())

        self.model.zero_grad()
        output[0, class_idx].backward()
        self.feature_maps[0], self.feature_maps[1] = (
            self.feature_maps[0 : len(self.target_layers[0])],
            self.feature_maps[len(self.target_layers[0]) :],
        )
        cams = [[], []]
        for i in range(len(self.target_layers)):
            for j in range(len(self.target_layers[i])):
                weights = self.grads[self.feature_maps[i][j][0]].mean(
                    dim=(-2, -1), keepdim=True
                )
                cams[i].append(
                    (weights * self.feature_maps[i][j][1]).sum(dim=1, keepdim=True)
                )
                cams[i][j] = nn.functional.relu(cams[i][j])

        return cams, class_idx #cams is [[sxcams],[mxcams]]


if __name__ == "__main__":
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()

    # 加载测试图像
    img_path = "test.jpg"
    img = Image.open(img_path).resize((224, 224), Image.ANTIALIAS)

    # 图像预处理
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

    # 最后，可以使用定义的 GradCAM 方法来生成 CAM 图像，并将 CAM 图像与原始图像叠加以便于可视化。

    # 定义 GradCAM 方法
    gradcam = GradCAM(model, model.layer4[2].conv3)

    # 生成 CAM 图像
    cam = gradcam(img_tensor)

    # 将 CAM 图像与原始图像叠加
    cam = nn.functional.interpolate(
        cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
    )
    cam = cam.cpu().detach().numpy()[0, 0]
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    result = cv2.addWeighted(
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.5, cam, 0.5, 0
    )

    cv2.imshow("gray_scale", result)
    cv2.waitKey(0)
    print("OVER")
