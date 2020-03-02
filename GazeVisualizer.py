import sys
import time
from _ctypes import Array
from typing import Dict, List
import cv2
import numpy as np
import torch
from torch import Tensor as Tensor
import torchvision.transforms as transforms
import imageio
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from GazeFrameRenderer import GazeFrameRenderer
from GazeExtractor import GazeExtractor

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
matplotlib.use('TkAgg')

IMAGE_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""

Generates images containing the gaze and head box of detected persons on videos or single images. Inherits GazeFrameRenderer for the graphical rendering, and is decorated 
by a GazeExtractor to obtain the gaze360 outputs and track people on input images. Transforms the extracted data to display them.

"""

class GazeVisualizer(GazeFrameRenderer):
    def __init__(self, gazeExtractor: GazeExtractor):
        super().__init__(gazeExtractor.width, gazeExtractor.height)
        self.extractor: GazeExtractor = gazeExtractor
        self.currentVideo: List = []
        self.instancesTracking = None
        self.W: int = 0
        self.color_encoding = [[random.randint(0, 254), random.randint(0, 254), random.randint(0, 254)] for i in
                               range(0, 1000)]

    def generateGazeVideoFromInput(self, filePath: str, outputPath: str, index: int = 0, offset: int = 0) -> None:
        reader = imageio.get_reader(filePath)
        fps = reader.get_meta_data()['fps']
        self.currentVideo = [reader.get_data(i) for i in range(index,index+offset)]
        self.W = max(int(fps // 8), 1)
        self.trackInstances()
        self.compileShader()
        out = imageio.get_writer(outputPath, fps=fps)
        for i in range(index, index + offset):
            print(i)
            f = self.generateFrame(i)
            out.append_data(f)
            # cv2.imwrite("/home/pandrieu/dev/tlab/test"+str(i)+".jpg", f[:, :, ::-1])
        out.close()

    def trackInstances(self):
        self.instancesTracking: Dict = self.extractor.getInstancesTracking(
            self.extractor.extractHeadsBoxesVideo(self.currentVideo))

    def generateFrame(self, i: int) -> Array:
        image = self.getResizedInput(i)
        if i in self.instancesTracking:
            for id_t in self.instancesTracking[i].keys():
                bbox, eyes = self.getBoxesEyes(i, id_t)
                image = self.getFullImage(image, self.getGazeArrow(eyes, self.getGazeFromInput(self.getInputImage(i, id_t))), bbox, id_t)
        return image.astype(np.uint8)

    def getResizedInput(self, i):
        return cv2.resize(self.currentVideo[i].copy(), (self.width, self.height)).astype(float)

    def getBoxesEyes(self, i, j):
        bbox, eyes = self.instancesTracking[i][j]
        bbox = np.asarray(bbox).astype(int)
        imageShape = self.currentVideo[i].shape
        dim0, dim1 = imageShape[0], imageShape[1]
        bbox[0], bbox[2], bbox[1], bbox[3] = self.width * bbox[0] / dim1, self.width * bbox[2] / dim1, self.height * \
                                             bbox[1] / dim0, self.height * bbox[3] / dim0
        eyes = np.asarray(eyes).astype(float)
        eyes[0], eyes[1] = eyes[0] / float(dim1), eyes[1] / float(dim0)
        return bbox, eyes

    def getInputImage(self, indexImage, indexInstance):
        input_image = torch.zeros(7, 3, 224, 224)
        count = 0
        for j in range(indexImage - 3 * self.W, indexImage + 4 * self.W, self.W):
            if j in self.instancesTracking and indexInstance in self.instancesTracking[j]:
                new_im = Image.fromarray(self.currentVideo[j], 'RGB')
                bbox, eyes = self.instancesTracking[j][indexInstance]
            else:
                new_im = Image.fromarray(self.currentVideo[indexImage], 'RGB')
                bbox, eyes = self.instancesTracking[indexImage][indexInstance]
            new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            input_image[count, :, :, :] = IMAGE_NORMALIZE(transforms.ToTensor()(transforms.Resize((224, 224))(new_im)))
            count += 1
        return input_image

    def getGazeFromInput(self, inputImage):
        return self.spherical2cartesial(self.extractor.getOutputGazeFromFrame(inputImage)).detach().numpy().reshape((-1))

    def getGazeArrow(self, eyes, gaze):
        return self.render_frame(2 * eyes[0] - 1, -2 * eyes[1] + 1, -gaze[0], gaze[1], -gaze[2], 0.05)

    def getFullImage(self, image, arrowRaw, bbox, id):
        binary_img = self.getBinaryImageArrow(arrowRaw)
        return cv2.rectangle((binary_img * image + arrowRaw * (1 - binary_img)).astype(np.uint8),
                             (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             self.color_encoding[min(id, 900)]).astype(float)

    def getBinaryImageArrow(self, imgArrow):
        binary_img = np.reshape(
            ((imgArrow[:, :, 0] + imgArrow[:, :, 1] + imgArrow[:, :, 2]) == 0.0).astype(float),
            (self.height, self.width, 1))
        return np.concatenate((binary_img, binary_img, binary_img), axis=2)

    @staticmethod
    def spherical2cartesial(x: Tensor) -> Tensor:
        output = torch.zeros(x.size(0), 3)
        output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
        output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
        output[:, 1] = torch.sin(x[:, 1])
        return output

def displayFrame(image) -> None:
    plt.figure(figsize=[12, 12])
    plt.imshow(image)
    plt.axis('off')
    plt.show()
