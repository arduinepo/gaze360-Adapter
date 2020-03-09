import sys
import time
from _ctypes import Array
from typing import Dict, List
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

    def generateGazeVideo(self, filePath: str, outputPath: str, index: int = 0, offset: int = 0) -> None:
        reader = imageio.get_reader(filePath)
        fps = reader.get_meta_data()['fps']
        self.extractor.W = max(int(fps // 8), 1)
        frames = [reader.get_data(i) for i in range(index, index + offset)]
        print("INfer and extract head boxes")
        instancesTracking = self.trackInstances(frames)
        print("Infer gazes")
        gazes = self.extractor.predictGazesFrames()
        self.compileShader()
        out = imageio.get_writer(outputPath, fps=fps)
        print("Render gazes")
        for i in range(0,offset):
            print(i)
            image = self.extractor.getResizedInput(frames[i])
            if i in instancesTracking:
                for id_t in instancesTracking[i].keys():
                    bbox, eyes = self.extractor.getBoxesEyes(i, id_t)
                    image = self.getFullImage(image, self.getGazeArrow(eyes, self.spherical2cartesial(gazes[i][id_t]).detach().numpy().reshape((-1))), bbox, id_t)
            out.append_data(image.astype(np.uint8))
            # cv2.imwrite("/home/pandrieu/dev/tlab/test"+str(i)+".jpg", f[:, :, ::-1])
        out.close()

    def addSingleGazeToFrame(self, gaze, frame, instance, currentImage):
        bbox, eyes = self.extractor.getBoxesEyes(frame, instance)
        gaze = self.spherical2cartesial(gaze).detach().numpy().reshape((-1))
        return self.getFullImage(currentImage, self.getGazeArrow(eyes, gaze), bbox, instance)

    def generateGazeVideoFromInput(self, filePath: str, outputPath: str, index: int = 0, offset: int = 0) -> None:
        reader = imageio.get_reader(filePath)
        fps = reader.get_meta_data()['fps']
        self.extractor.W = max(int(fps // 8), 1)
        currentVideo = [reader.get_data(i) for i in range(index, index + offset)]
        instancesTracking = self.trackInstances(currentVideo)
        self.compileShader()
        out = imageio.get_writer(outputPath, fps=fps)
        for i in range(0, offset):
            print("generates:",i)
            image = self.extractor.getResizedInput(currentVideo[i])
            if i in instancesTracking:
                for id_t in instancesTracking[i].keys():
                    bbox, eyes = self.extractor.getBoxesEyes(i, id_t)
                    image = self.getFullImage(image, self.getGazeArrow(eyes, self.getGazeFromInput(
                        self.extractor.getInputImage(i, id_t))), bbox, id_t)
            out.append_data(image.astype(np.uint8))
            # cv2.imwrite("/home/pandrieu/dev/tlab/test"+str(i)+".jpg", f[:, :, ::-1])
        out.close()

    def trackInstances(self, video):
        return self.extractor.getInstancesTracking(self.extractor.extractHeadsBoxesVideo(video))

    def getGazeFromInput(self, inputImage):
        return self.spherical2cartesial(self.extractor.getOutputGazeFromFrame(inputImage)).detach().numpy().reshape(
            (-1))

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
