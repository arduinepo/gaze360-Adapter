import sys, os
from _ctypes import Array
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import time
import cv2
from PIL import Image
from GazePredictor import GazePredictor
import torchvision.transforms as transforms

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose.vis.extractor import extract_boxes_xywh_from_instances, DensePoseResultExtractor
from densepose.structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from PosePredictor import PosePredictor, PosePredictorMultiGPU

POSE_EXTRACTOR = DensePoseResultExtractor()
IMAGE_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""




"""

class GazeExtractor:
    def __init__(self, configFile: str, poseModelFile: str, gazeModelFile: str, width: int,
                 height: int):
        self.posePredictor = PosePredictorMultiGPU(configFile, poseModelFile)
        self.gazePredictor = GazePredictor(gazeModelFile)
        self.height, self.width = height, width
        self.W: int = 0
        self.currentVideo = None
        self.instancesTracking = None

    def extractInstancesFromOutputs(self, outputs) -> List:
        globalInstances = []
        for output in outputs:
            boxes = output[0]
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            listBoxes = boxes.tolist()
            resultInstances = []
            for j, box in enumerate(listBoxes):
                w, h = int(box[2]), int(box[3])
                result = torch.zeros([1, h, w], dtype=torch.uint8, device="cpu")
                S = F.interpolate(output[1][[j]], (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
                result[0] = (F.interpolate(output[2][[j]], (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
                             * (S > 0).long()).squeeze(0)
                resultInstances.append([boxes[j], result.numpy()[0]])
            globalInstances.append(resultInstances)
        return globalInstances

    def predictGazesFrames(self):
        """gazesList = []
        for i in range(min(self.instancesTracking.keys()), max(self.instancesTracking.keys())):
            if i in self.instancesTracking:
                for id_t in self.instancesTracking[i].keys():
                    gaze, _ = self.gazePredictor(self.getInputImage(i, id_t))
                    gazesList.append(gaze)"""
        inputs = []
        print("input transformation for gaze inference")
        for i in range(min(self.instancesTracking.keys()), max(self.instancesTracking.keys()) + 1):
            print(i)
            if i in self.instancesTracking:
                for id_t in self.instancesTracking[i].keys():
                    inputs.append(self.getInputImage(i, id_t))
        gazesList = self.gazePredictor.predictMultipleInputs(inputs)
        gazesByFrame = dict()
        rank = 0
        for i in range(min(self.instancesTracking.keys()), max(self.instancesTracking.keys()) + 1):
            frame = dict()
            if i in self.instancesTracking:
                for j in self.instancesTracking[i].keys():
                    frame[j] = gazesList[rank]
                    rank += 1
                gazesByFrame[i] = frame
        return gazesByFrame

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
            im = transforms.Resize((224, 224))(new_im)
            input_image[count, :, :, :] = IMAGE_NORMALIZE(transforms.ToTensor()(im))
            count += 1
        return input_image.view(1, 7, 3, 224, 224)

    def predictGazesMultipleImages(self, images):
        globalOutputs = []
        batches = []
        numberImages = len(images)
        batchSize = 20
        batchNumber = int(round(numberImages / batchSize))
        if batchNumber < numberImages / batchSize:
            batchNumber += 1
        underTotal = 0
        with torch.no_grad():
            for j in range(batchNumber):
                batch = []
                for i in range(0, min(batchSize, numberImages - underTotal)):
                    batch.append(images[j * batchSize + i].view(1, 7, 3, 224, 224))
                batches.append(batch)
                underTotal += batchSize
            for listFrames in batches:
                # TODO : transformation préalable
                for output in self.gazePredictor(listFrames):
                    globalOutputs.append(output)
        return globalOutputs

    def predictGazesSingleImage(self, image):
        self.getInstancesTracking(self.extractHeadsBoxesImage(image))
        outputs = []
        for id_t in self.instancesTracking[0].keys():
            outputs.append(self.getOutputGazeFromFrame(self.getInputImage(0, id_t)))
        return outputs

    def extractHeadsBoxesVideo(self, frames: List) -> Dict:
        final_results = dict()
        self.currentVideo = frames
        print("Inference")
        outputs = self.posePredictor(frames)
        print("Extraction")
        results = self.extractInstancesFromOutputs(outputs)
        for i in range(0, len(frames)):
            headsBoxes = []
            for instance in results[i]:
                headBox = self.extract_head_bbox(instance)
                if len(headBox) > 0:
                    headsBoxes.append(headBox)
            if len(headsBoxes) > 0:
                final_results[i] = headsBoxes
        return final_results

    def extractHeadsBoxesImage(self, image):
        outputs = self.extractInstancesFromFrame(image)
        res = dict()
        headsBoxes = []
        for instance in outputs:
            headBox = self.extract_head_bbox(instance)
            if len(headBox) > 0:
                headsBoxes.append(headBox)
        res[0] = headsBoxes
        return res

    def getOutputGazeFromFrame(self, image) -> torch.Tensor:
        out, _ = self.gazePredictor(image.cuda())
        return out

    def getInstancesTracking(self, listsHeadsBoxes):
        id_num = 0
        tracking_id = dict()
        identity_last = dict()
        frames_with_people = list(listsHeadsBoxes.keys())
        frames_with_people.sort()
        for i in frames_with_people:
            speople = listsHeadsBoxes[i]
            identity_next = dict()
            for j in range(len(speople)):
                bbox_head = speople[j]
                if bbox_head is None:
                    continue
                id_val = self.getInstanceFromPreviousFrame(bbox_head, identity_last)
                if id_val is None:
                    id_num += 1
                    id_val = id_num
                # TODO: Improve eye location
                eyes = [(bbox_head[0] + bbox_head[2]) / 2.0, (0.65 * bbox_head[1] + 0.35 * bbox_head[3])]
                identity_next[id_val] = (bbox_head, eyes)
            tracking_id[i] = identity_last = identity_next
        self.instancesTracking = tracking_id
        return tracking_id

    # renvoie clé de id_dict dont la valeur box recoupe le plus la box en paramètre
    def getInstanceFromPreviousFrame(self, bbox, id_dict) -> int:
        id_final = None
        max_iou = 0.5
        for k in id_dict.keys():
            iou = self.computeInterOverUnion(bbox, id_dict[k][0])
            if iou > max_iou:
                id_final = k
                max_iou = iou
        return id_final

    def extractInstancesFromFrame(self, image) -> List:
        outputs = self.posePredictor.predictOneInput(image)
        result = POSE_EXTRACTOR(outputs).results
        boxes = extract_boxes_xywh_from_instances(outputs)
        resultInstances = []
        for j in range(0, len(outputs)):
            iuv_arr = DensePoseResult.decode_png_data(*result[j])[0]
            box = boxes[j]
            resultInstances.append([box, iuv_arr])
        return resultInstances

    def extract_head_bbox(self, instance):
        box = instance[0]
        iuv_mask_head = instance[1][:, :] > 22
        bbox = []
        if iuv_mask_head.any():
            bbox = self.mask_to_bbox(iuv_mask_head)
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] + box[0], bbox[1] + box[1], bbox[2] + box[0], bbox[3] + box[1]
        return bbox

    """
    Computes Intersection Over Union (IOU) ratio of two boxes
    """

    @staticmethod
    def computeInterOverUnion(bb1, bb2) -> float:
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        eps = 1e-8
        if iou <= 0.0 or iou > 1.0 + eps:
            return 0.0
        return iou

    @staticmethod
    def mask_to_bbox(mask) -> np.ndarray:
        # Code from Detectron modifyied to make the box 30% bigger
        """Compute the tight bounding box of a binary mask."""
        xs = np.where(np.sum(mask, axis=0) > 0)[0]
        ys = np.where(np.sum(mask, axis=1) > 0)[0]
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0 = xs[0]
        x1 = xs[-1]
        y0 = ys[0]
        y1 = ys[-1]
        w = x1 - x0
        h = y1 - y0
        x0 = max(0, x0 - w * 0.15)
        x1 = max(0, x1 + w * 0.15)
        y0 = max(0, y0 - h * 0.15)
        y1 = max(0, y1 + h * 0.15)
        return np.array((x0, y0, x1, y1), dtype=np.float32)

    def getResizedInput(self, image):
        return cv2.resize(image.copy(), (self.width, self.height)).astype(float)

    def getBoxesEyes(self, i, j):
        bbox, eyes = self.instancesTracking[i][j]
        bbox = np.asarray(bbox).astype(int)
        imageShape = self.currentVideo[i].shape
        dim0, dim1 = imageShape[0:2]
        bbox[0], bbox[2], bbox[1], bbox[3] = self.width * bbox[0] / dim1, self.width * bbox[2] / dim1, self.height * \
                                             bbox[1] / dim0, self.height * bbox[3] / dim0
        eyes = np.asarray(eyes).astype(float)
        eyes[0], eyes[1] = eyes[0] / float(dim1), eyes[1] / float(dim0)
        return bbox, eyes
