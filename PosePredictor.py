# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import sys
import time
import detectron2.data.transforms as T
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import build_model
import GazeData

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose import add_densepose_config

class PosePredictor:
    def __init__(self, cfg, gpu):
        self.cfg = cfg
        self.gpu = gpu
        self.model = build_model(cfg).cuda(gpu)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images):
        globalOutputs = []
        dataLoader = GazeData.DensePoseDataLoader(original_images, self.cfg, self.gpu, batchSize=3)
        for batch in enumerate(dataLoader):
            outputs = self.model(batch[1])
            for output in outputs:
                out = output['instances']
                globalOutputs.append(extractResults(out))
                del out
            torch.cuda.empty_cache()
        return globalOutputs

class PosePredictorMultiGPU:
    def __init__(self, configFile, poseModelFile, accuracyThreshold,numberGpus):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(configFile)
        cfg.MODEL.WEIGHTS = poseModelFile
        cfg.freeze()
        self.cfg = cfg
        self.accuracyThreshold = accuracyThreshold
        mp.set_start_method("spawn", force=True)
        smp = mp.get_context("spawn")
        self.queues = [smp.Manager().list(), smp.Manager().list()]
        self.procs = []
        for i in range(numberGpus):
            cfgProc = cfg.clone()
            cfgProc.defrost()
            cfgProc.MODEL.DEVICE = "cuda:" + str(i)
            self.procs.append(PosePredictorMultiGPU.PredictWorker(cfgProc, self.queues[i], i, self.accuracyThreshold))

    class PredictWorker(mp.Process):
        def __init__(self, cfg, queue, i, accuracyThreshold):
            self.predictor = None
            self.cfg = cfg
            self.resultQueue = queue
            self.inputs = None
            self.nProc = i
            self.accuracyThreshold = accuracyThreshold
            super().__init__()

        def run(self):
            self.predictor = PosePredictor(self.cfg, self.nProc)
            dataLoader = GazeData.DensePoseDataLoader(self.inputs, self.cfg, self.nProc, batchSize=3)
            for batch in enumerate(dataLoader):
                outputs = self.predictor.model(batch[1])
                for output in outputs:
                    out = output['instances']
                    result=self.extractResults(out)
                    if len(result) > 0:
                        self.resultQueue.append(result)
                    del out
                torch.cuda.empty_cache()

        def extractResults(self,output):
            boxes = output.pred_boxes.tensor.detach().to("cpu")
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            listBoxes = boxes.tolist()
            resultInstances = []
            for j, box in enumerate(listBoxes):
                if output.scores[j] > self.accuracyThreshold:
                    w, h = int(box[2]), int(box[3])
                    S = F.interpolate(output.pred_densepose.S.detach()[[j]], (h, w), mode="bilinear",
                                      align_corners=False).argmax(dim=1)
                    result = (F.interpolate(output.pred_densepose.I.detach()[[j]], (h, w), mode="bilinear",
                                            align_corners=False).argmax(
                        dim=1) * (S > 0).long()).squeeze(0)
                    resultInstances.append([boxes[j], result.to("cpu").numpy()])
            return resultInstances

    def __call__(self, images):
        median = int(round(len(images) / 2))
        batchesGpus = [images[0:median], images[median:]]
        i = 0
        for p in self.procs:
            p.inputs = batchesGpus[i]
            p.start()
            i += 1
        for p in self.procs:
            p.join()
        outputs = []
        for queue in self.queues:
            for output in queue:
                outputs.append(output)
        return outputs
