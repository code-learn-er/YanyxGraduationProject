import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2


def get_data(root_dir):
    tongue_path = os.path.join(root_dir, "GXY_tongue")
    eyes_path = os.path.join(root_dir, "GXY-eyes")
    tongue = [os.path.join(tongue_path, i) for i in os.listdir(tongue_path)]
    eyes = [os.path.join(eyes_path, i) for i in os.listdir(eyes_path)]

    tongue = [os.path.join(i, j) for i in tongue for j in os.listdir(i)]
    eyes = [os.path.join(i, j) for i in eyes for j in os.listdir(i)]

    tongue = [os.path.join(i, j) for i in tongue for j in os.listdir(i)]
    eyes = [os.path.join(i, j) for i in eyes for j in os.listdir(i)]

    tongue.sort(key=lambda x: x.rsplit("/")[-1])
    eyes.sort(key=lambda x: x.rsplit("/")[-1])
    return list(zip(tongue, eyes))

label={"Feiyinxu": torch.tensor(0), "Yinxu": torch.tensor(1)}
class MyDataset(Dataset):
    def __init__(self, x, train=True, transform=None):
        super().__init__()
        self.data = x
        self.train = train
        self.transform = transform
        self.label = label
        # t=[i for i in os.listdir(root_dir) if os.path.isdir(i)]

    def __getitem__(self, index):
        tonguepath, eyespath = self.data[index]

        label = os.path.split(os.path.split(tonguepath)[0])[1]
        tongue = np.array(Image.open(tonguepath))
        eyes = [
            np.array(Image.open(os.path.join(eyespath, i)))
            for i in os.listdir(eyespath)
        ]
        if self.transform is not None:
            tongue = self.transform(tongue)
            eyes = [self.transform(i) for i in eyes]
        # cv2.imshow("test", tongue)
        return [tongue]+eyes, self.label[label]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    data = get_data(
        r"/media/codelearner/E2EE175BEE1726F7/Users/QuickLearner/Documents/python/graduationProject/data"
    )
    a = MyDataset(
        x=data,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        ),
    )
    cnt=1
    for (tongue, eyes), label in a:
        if len(eyes)!=10:
            print(cnt)
        cnt+=1
        # cv2.imshow("gray_scale", eyes[0])
        # cv2.waitKey(0)
        # print(label)
