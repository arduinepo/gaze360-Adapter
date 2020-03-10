import sys
import time
from typing import Dict, List
import numpy as np
import torch
from GazeData import GazeDataLoader, GazeDataSet
from GazePredictor import GazePredictor

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose.vis.extractor import extract_boxes_xywh_from_instances, DensePoseResultExtractor
from densepose.structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from PosePredictor import PosePredictor, PosePredictorMultiGPU

"""




"""

class GazeExtractor:
    def __init__(self, configFile: str, poseModelFile: str, gazeModelFile: str, width: int,
                 height: int, poseAccuracyThreshold, numberGpus):
        self.posePredictor = PosePredictorMultiGPU(configFile, poseModelFile, poseAccuracyThreshold, numberGpus)
        self.gazePredictor = GazePredictor(gazeModelFile)
        self.height, self.width = height, width

    def extractGazeFromVideo(self, frames, fps):
        W = max(int(fps // 8), 1)
        print("heads boxes extraction")
        t = time.time()
        instancesTracking = self.getInstancesTracking(self.extractHeadsBoxesVideo(frames))
        print("infer gazes")
        gazesList = self.predictGazes(GazeDataLoader(GazeDataSet(frames, instancesTracking, W)))
        print("maps gazes with frames")
        gazes = self.buildMapFramesGazes(gazesList, instancesTracking)
        print(time.time() - t)
        return gazes, instancesTracking

    def extractHeadsBoxesVideo(self, frames: List) -> Dict:
        headsBoxesByFrame = dict()
        print("detect individuals")
        outputs = self.posePredictor(frames)
        for i in range(0, len(frames)):
            headsBoxes = []
            for instance in outputs[i]:
                headBox = self.extract_head_bbox(instance)
                if len(headBox) > 0:
                    headsBoxes.append(headBox)
            if len(headsBoxes) > 0:
                headsBoxesByFrame[i] = headsBoxes
        return headsBoxesByFrame

    def getInstancesTracking(self, listsHeadsBoxes):
        print("track instances")
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
        return tracking_id

    def predictGazes(self, dataLoader):
        print("starts gaze prediction")
        gazesList = []
        for data in enumerate(dataLoader):
            out = self.gazePredictor(data)
            for o in out:
                gazesList.append(o.detach())
            del out
            torch.cuda.empty_cache()
        return gazesList

    @staticmethod
    def buildMapFramesGazes(gazesList, instances):
        gazesByFrame = dict()
        rank = 0
        for i in range(min(instances.keys()), max(instances.keys()) + 1):
            frame = dict()
            if i in instances:
                for j in instances[i].keys():
                    frame[j] = gazesList[rank]
                    rank += 1
                gazesByFrame[i] = frame
        return gazesByFrame

    def extract_head_bbox(self, instance):
        box = instance[0]
        iuv_mask_head = instance[1][:, :] > 22
        bbox = []
        if iuv_mask_head.any():
            bbox = self.mask_to_bbox(iuv_mask_head)
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] + box[0], bbox[1] + box[1], bbox[2] + box[0], bbox[3] + box[1]
        return bbox

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

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def displayFrame(image) -> None:
    plt.figure(figsize=[12, 12])
    plt.imshow(image)
    plt.axis('off')
    plt.show()
