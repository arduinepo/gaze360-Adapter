import sys
from _ctypes import Array
from typing import Dict, List, Tuple
import numpy as np
import torch
import time
from detectron2.config import get_cfg

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose.vis.extractor import extract_boxes_xywh_from_instances, DensePoseResultExtractor
from densepose.structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from densepose import add_densepose_config
from gaze360.code.model import GazeLSTM
from PosePredictor import PosePredictor

POSE_EXTRACTOR = DensePoseResultExtractor()
GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""



"""

class GazeExtractor:
    def __init__(self, configFile: str, poseModelFile: str, gazeModelFile: str, numberGPUs: int, width: int,
                 height: int):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(configFile)
        cfg.MODEL.WEIGHTS = poseModelFile
        cfg.freeze()
        self.posePredictor = PosePredictor(cfg)
        self.gazePredictor = torch.nn.DataParallel(GazeLSTM()).cuda()
        checkpoint = torch.load(gazeModelFile)
        self.gazePredictor.load_state_dict(checkpoint['state_dict'])
        self.gazePredictor.eval()
        self.height, self.width = height, width

    def getGazesFromImage(self, image):
        instancesTracking = self.getInstancesTracking(self.extractHeadsBoxesImage(image))
        print()


    def getOutputGazeFromFrame(self, image) -> torch.Tensor:
        out, _ = self.gazePredictor(image.view(1, 7, 3, 224, 224).cuda())
        return out

    def predictGazesMultipleFrames(self, images):
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
                for output in self.gazePredictor(listFrames):
                    globalOutputs.append(output)
        return globalOutputs

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

    def extractHeadsBoxesImage(self, image):
        outputs = self.extractInstancesFromFrame(image)
        headsBoxes = []
        for instance in outputs:
            headBox = self.extract_head_bbox(instance)
            if len(headBox) > 0:
                headsBoxes.append(headBox)
        return headsBoxes

    def extractHeadsBoxesVideo(self, frames: List) -> Dict:
        final_results = dict()
        outputs = self.extractInstancesFromFrames(frames)
        t = time.time()
        for i in range(0, len(frames)):
            headsBoxes = []
            for instance in outputs[i]:
                headBox = self.extract_head_bbox(instance)
                if len(headBox) > 0:
                    headsBoxes.append(headBox)
            if len(headsBoxes) > 0:
                final_results[i] = headsBoxes
        print("durée moyenne d'extraction :", (time.time() - t) / len(frames))
        return final_results

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

    def extractInstancesFromFrames(self, images) -> List:
        outputs = self.posePredictor(images)
        globalInstances = []
        t = time.time()
        for output in outputs:
            result = POSE_EXTRACTOR(output).results
            boxes = extract_boxes_xywh_from_instances(output)
            resultInstances = []
            for j in range(0, len(output)):
                iuv_arr = DensePoseResult.decode_png_data(*result[j])[0]
                box = boxes[j]
                resultInstances.append([box, iuv_arr])
            globalInstances.append(resultInstances)
        print((time.time() - t) / len(outputs))
        return globalInstances

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
