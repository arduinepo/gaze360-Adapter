import sys
import cv2
import numpy as np
import torch
from torch import Tensor as Tensor
import imageio
import random
import matplotlib
import matplotlib.pyplot as plt
from GazeFrameRenderer import GazeFrameRenderer
from GazeExtractor import GazeExtractor

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
matplotlib.use('TkAgg')

"""

Generates images containing the gaze and head box of detected persons on videos or single images. Inherits GazeFrameRenderer for the graphical rendering, and is decorated 
by a GazeExtractor to obtain the gaze360 outputs and track people on input images. Transforms the extracted data to display them.

"""

class GazeVisualizer(GazeFrameRenderer):
    def __init__(self, gazeExtractor: GazeExtractor):
        super().__init__(gazeExtractor.width, gazeExtractor.height)
        self.extractor: GazeExtractor = gazeExtractor
        self.color_encoding = [[random.randint(0, 254), random.randint(0, 254), random.randint(0, 254)] for i in
                               range(0, 1000)]
        self.frames = None
        self.instances = None

    def generateGazeVideo(self, filePath: str, outputPath: str, index: int = 0, offset: int = 0) -> None:
        reader = imageio.get_reader(filePath)
        fps = reader.get_meta_data()['fps']
        self.frames = [reader.get_data(i) for i in range(index, index + offset)]
        gazes, self.instances = self.extractor.extractGazeFromVideo(self.frames, fps)
        self.compileShader()
        out = imageio.get_writer(outputPath, fps=fps)
        print("write frames")
        for i in range(0, offset):
            if i % 10 == 0:
                print("frame", i)
            image = self.getResizedInput(self.frames[i])
            if i in self.instances:
                for id_t in self.instances[i].keys():
                    bbox, eyes = self.getBoxesEyes(i, id_t)
                    image = self.getFullImage(image, self.getGazeArrow(eyes, self.spherical2cartesial(
                        gazes[i][id_t]).detach().numpy().reshape((-1))), bbox, id_t)
            out.append_data(image.astype(np.uint8))
        out.close()

    def addSingleGazeToFrame(self, gaze, frame, instance, currentImage):
        bbox, eyes = self.getBoxesEyes(frame, instance)
        gaze = self.spherical2cartesial(gaze).detach().numpy().reshape((-1))
        return self.getFullImage(currentImage, self.getGazeArrow(eyes, gaze), bbox, instance)

    def getResizedInput(self, image):
        return cv2.resize(image.copy(), (self.width, self.height)).astype(float)

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

    def getBoxesEyes(self, i, j):
        bbox, eyes = self.instances[i][j]
        bbox = np.asarray(bbox).astype(int)
        imageShape = self.frames[i].shape
        dim0, dim1 = imageShape[0:2]
        bbox[0], bbox[2], bbox[1], bbox[3] = self.width * bbox[0] / dim1, self.width * bbox[2] / dim1, self.height * \
                                             bbox[1] / dim0, self.height * bbox[3] / dim0
        eyes = np.asarray(eyes).astype(float)
        eyes[0], eyes[1] = eyes[0] / float(dim1), eyes[1] / float(dim0)
        return bbox, eyes

    @staticmethod
    def spherical2cartesial(x: Tensor) -> Tensor:
        """output = torch.zeros(x.size(0), 3)
        output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
        output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
        output[:, 1] = torch.sin(x[:, 1])"""
        output = torch.zeros(3)
        output[2] = -torch.cos(x[1]) * torch.cos(x[0])
        output[0] = torch.cos(x[1]) * torch.sin(x[0])
        output[1] = torch.sin(x[1])
        return output

def displayFrame(image) -> None:
    plt.figure(figsize=[12, 12])
    plt.imshow(image)
    plt.axis('off')
    plt.show()
