import torch
from torch import nn
from resnet import Bottleneck, ResNet


def get_feature_classifier(model):
    features = nn.Sequential(
        *[
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        ]
    )
    classifier = nn.Sequential(
        *[
            model.layer4,
        ]
    )
    return features, classifier


def jointImages(images):
    mx1 = torch.cat(tuple(images[0:5]), dim=3)
    mx2 = torch.cat(tuple(images[5:]), dim=3)
    mx = torch.cat((mx1, mx2), dim=2)
    return mx


class MultResNet(nn.Module):
    def __init__(self, sx_model_index, mx_model_index):
        super().__init__()
        model1 = ResNet(Bottleneck, sx_model_index)
        model2 = ResNet(Bottleneck, mx_model_index)
        self.features1, self.classifier1 = get_feature_classifier(model1)
        self.features2, self.classifier2 = get_feature_classifier(model2)

        self.features = [
            self.features1,
            self.features2,
        ]
        self.classifier = [
            self.classifier1,
            self.classifier2,
        ]
        
        self.sxfc = self.getClassifier()
        self.mxfc0 = self.getClassifier()
        self.mxfc1 = self.getClassifier()
        self.mxfc2 = self.getClassifier()
        self.mxfc3 = self.getClassifier()
        self.mxfc4 = self.getClassifier()
        self.mxfc5 = self.getClassifier()
        self.mxfc6 = self.getClassifier()
        self.mxfc7 = self.getClassifier()
        self.mxfc8 = self.getClassifier()
        self.mxfc9 = self.getClassifier()
        self.mxfc = [
            self.mxfc0,
            self.mxfc1,
            self.mxfc2,
            self.mxfc3,
            self.mxfc4,
            self.mxfc5,
            self.mxfc6,
            self.mxfc7,
            self.mxfc8,
            self.mxfc9,
        ]

    def forward(self, x):
        xx = [x[0], jointImages(x[1:])]

        out = []
        for i in range(len(self.features)):
            out.append(self.features[i](xx[i]))
            out[i] = self.classifier[i](out[i])

        feature_size = out[0].shape[-2:]
        out[0] = nn.functional.avg_pool2d(out[0], feature_size)
        out[1] = nn.functional.avg_pool2d(out[1], feature_size, feature_size)

        out[0] = out[0].view(out[0].shape[0], -1)
        out[0] = self.sxfc(out[0])
        out[1] = out[1].view(out[1].shape[0], out[1].shape[1], -1)
        out[1] = torch.chunk(out[1], 10, 2)

        out[1] = [self.mxfc[i](out[1][i].view(out[0].shape[0], -1)) for i in range(len(out[1]))]
        out[1] = sum(out[1])
        out = 5 * out[0] + out[1]
        return out
    
    def getClassifier(self):
        return nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 2),
        )


class MultCNN(nn.Module):
    def __init__(self, *index):
        super(MultCNN, self).__init__()
        self.sxcnn = self.VGGPipeline()
        self.mxcnn = self.VGGPipeline()
        self.sxfc = self.getClassifier()
        self.mxfc0 = self.getClassifier()
        self.mxfc1 = self.getClassifier()
        self.mxfc2 = self.getClassifier()
        self.mxfc3 = self.getClassifier()
        self.mxfc4 = self.getClassifier()
        self.mxfc5 = self.getClassifier()
        self.mxfc6 = self.getClassifier()
        self.mxfc7 = self.getClassifier()
        self.mxfc8 = self.getClassifier()
        self.mxfc9 = self.getClassifier()
        self.mxfc = [
            self.mxfc0,
            self.mxfc1,
            self.mxfc2,
            self.mxfc3,
            self.mxfc4,
            self.mxfc5,
            self.mxfc6,
            self.mxfc7,
            self.mxfc8,
            self.mxfc9,
        ]

    def forward(self, x):
        xx = [x[0], jointImages(x[1:])]

        out = []
        out.append(self.sxcnn(xx[0]))
        out.append(self.mxcnn(xx[1]))

        feature_size = out[0].shape[-2:]
        out[0] = nn.functional.avg_pool2d(out[0], feature_size)
        out[1] = nn.functional.avg_pool2d(out[1], feature_size, feature_size)

        out[0] = out[0].view(out[0].shape[0], -1)
        out[0] = self.sxfc(out[0])
        out[1] = out[1].view(out[1].shape[0], out[1].shape[1], -1)
        out[1] = torch.chunk(out[1], 10, 2)

        out[1] = [self.mxfc[i](out[1][i].view(out[0].shape[0], -1)) for i in range(len(out[1]))]
        out[1] = sum(out[1])
        out = out[0] + out[1]
        return out

    def getClassifier(self):
        return nn.Sequential(
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def VGGPipeline(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


model = {"MultResNet": MultResNet, "MultCNN": MultCNN}
