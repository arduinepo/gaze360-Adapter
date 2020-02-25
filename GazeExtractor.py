import sys
from _ctypes import Array
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.nn.functional import interpolate
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose.vis.extractor import extract_boxes_xywh_from_instances, DensePoseResultExtractor
from densepose.structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from densepose import add_densepose_config
from gaze360.code.model import GazeLSTM

POSE_EXTRACTOR = DensePoseResultExtractor()

"""



"""

class GazeExtractor:
    def __init__(self, configFile: str, poseModelFile: str, gazeModelFile: str, width: int, height: int):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(configFile)
        cfg.MODEL.WEIGHTS = poseModelFile
        cfg.freeze()
        self.posePredictor = DefaultPredictor(cfg)
        self.gazePredictor = torch.nn.DataParallel(GazeLSTM()).cuda()
        self.gazePredictor.cuda()
        checkpoint = torch.load(gazeModelFile)
        self.gazePredictor.load_state_dict(checkpoint['state_dict'])
        self.gazePredictor.eval()
        self.height, self.width = height, width

    def getOutputGaze(self, image) -> torch.Tensor:
        out, _ = self.gazePredictor(image.view(1, 7, 3, 224, 224).cuda())
        return out

    def getInstancesTracking(self, headsBoxes: Dict) -> Dict:
        id_num = 0
        tracking_id = dict()
        identity_last = dict()
        frames_with_people = list(headsBoxes.keys())
        frames_with_people.sort()
        for i in frames_with_people:
            # box globale courante
            speople = headsBoxes[i]
            # map des individus
            identity_next = dict()
            for j in range(len(speople)):
                # box individuelle courante
                bbox_head = speople[j]
                if bbox_head is None:
                    continue
                id_val = self.find_id(bbox_head, identity_last)
                if id_val is None:
                    id_num += 1
                    id_val = id_num
                # TODO: Improve eye location
                eyes = [(bbox_head[0] + bbox_head[2]) / 2.0, (0.65 * bbox_head[1] + 0.35 * bbox_head[3])]
                identity_next[id_val] = (bbox_head, eyes)
            identity_last = identity_next
            tracking_id[i] = identity_last
        return tracking_id

    def extractHeadsBoxesVideo(self, frames: List, index: int = 0, offset: int = 0) -> Dict:
        final_results = dict()
        for i in range(index, offset):
            img = frames[i].copy()
            iuv, inds = self.extractIuvIndsFromFrame(img)
            bbox = self.extract_heads_bbox(iuv, inds)
            self.printList(bbox)
            if len(bbox) > 0:
                final_results[i] = bbox
        return final_results

    def extractIuvIndsFromFrame(self, image: Array) -> Tuple:
        outputs = self.posePredictor(image)['instances']
        result = POSE_EXTRACTOR(outputs).results
        boxes = extract_boxes_xywh_from_instances(outputs)
        segm = outputs.pred_densepose.S
        h, w = image.shape[0], image.shape[1]
        iuv = np.zeros((h, w), dtype=int)
        inds = np.zeros((h, w), dtype=int)
        for j in range(0, len(outputs)):
            iuv_arr = DensePoseResult.decode_png_data(*result[j])[0]
            box = boxes[j]
            x, y, w, h = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            inds_arr = interpolate(segm[[j]], (h, w), mode="bilinear", align_corners=False).argmax( dim=1).cpu().numpy()[0]
            for u in range(x, x + w):
                for t in range(y, y + h):
                    inds[t][u] = inds_arr[t - y][u - x]
                    iuv[t][u] = iuv_arr[t - y][u - x]
        return iuv, inds

    def extract_heads_bbox(self, iuv, inds) -> List:
        # masque têtes
        iuv_mask_head = iuv[:, :] > 22
        # nombre de masques de parties corporelles
        N = inds[:].max()
        if N == 0:
            if ~iuv_mask_head.any():
                return []
            else:
                bbox = self.mask_to_bbox(iuv_mask_head.astype(np.uint))
                return [self.mask_to_bbox(iuv_mask_head.astype(np.uint))]
        bbox_list = []
        for i in range(N, 0, -1):
            # masque de la partie corporelle en cours
            mask_id = inds[:, :] == i
            # intersection masque têtes et masque courant
            mask_head_person = iuv_mask_head & mask_id
            if mask_head_person.any():
                # si intersection non nulle, contient tête : box de la tête
                bbox = self.mask_to_bbox(mask_head_person)
                bbox_list.append(bbox)
            # soustraction masque têtes - partie courante
            iuv_mask_head = iuv_mask_head & (~mask_head_person)
        if iuv_mask_head.any():
            # si reste non nul, box
            bbox = self.mask_to_bbox(iuv_mask_head).astype(np.uint)
            bbox_list.append(bbox)
        return bbox_list

    @staticmethod
    def printArray(array):
        print(array.shape)
        if array[0] is Array:
            if array[0][0] is Array:
                print(array[0][0][0])
            else:
                print(array[0][0])
        else:
            print(array[0])

    @staticmethod
    def printList(list):
        print(len(list))
        for item in list:
            print(item)

    def find_id(self, bbox, id_dict) -> int:
        id_final = None
        max_iou = 0.5
        for k in id_dict.keys():
            iou = self.computeIOU(bbox, id_dict[k][0])
            if iou > max_iou:
                id_final = k
                max_iou = iou
        return id_final

    """
    Computes Intersection Over Union (IOU) of two boxes
    """
    @staticmethod
    def computeIOU(bb1, bb2) -> float:
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        print("intersection : ",intersection_area)
        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        print("boxes areas : ",bb1_area,bb2_area)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        print("iou : ",iou)
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
        W = mask.shape[0]
        H = mask.shape[1]
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
