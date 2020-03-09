# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import os, sys, time
from typing import List
import torch
import torch.multiprocessing as mp
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from torch.utils.data import DataLoader
import detectron2.data.transforms as T
from GazeData import GazeDataLoader

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
from densepose import add_densepose_config

class PosePredictor:

    def __init__(self, cfg, gpu):
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
        listsFrames = []
        numberImages = len(original_images)
        batchSize = 20
        batchNumber = int(round(numberImages / batchSize))
        if batchNumber < numberImages / batchSize:
            batchNumber += 1
        underTotal = 0
        with torch.no_grad():
            for j in range(batchNumber):
                list = []
                for i in range(0, min(batchSize, numberImages - underTotal)):
                    img = original_images[j * batchSize + i]
                    if self.input_format == "RGB":
                        img = img[:, :, ::-1]
                    height, width = img.shape[:2]
                    image = self.transform_gen.get_transform(img).apply_image(img)
                    list.append({"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda(),
                                 "height": height, "width": width})
                listsFrames.append(list)
                underTotal += batchSize
            for listFrames in listsFrames:
                outputs = self.model(listFrames)
                for output in outputs:
                    out = output['instances']
                    globalOutputs.append(out.to("cpu"))
                    del out
                    torch.cuda.empty_cache()
        return globalOutputs

    def predictOneInput(self, image):
        with torch.no_grad():
            if self.input_format == "RGB":
                img = image[:, :, ::-1]
            height, width = img.shape[:2]
            img = self.transform_gen.get_transform(img).apply_image(img)
            img = {"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).cuda(),
                   "height": height, "width": width}
            return self.model([img])[0]

class PosePredictorMultiGPU:
    def __init__(self, configFile, poseModelFile):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(configFile)
        cfg.MODEL.WEIGHTS = poseModelFile
        cfg.freeze()
        self.cfg = cfg
        mp.set_start_method("spawn", force=True)
        smp = mp.get_context("spawn")
        self.resultQueue = smp.Manager().list()
        self.procs = []
        for i in range(2):
            cfgProc = cfg.clone()
            cfgProc.defrost()
            cfgProc.MODEL.DEVICE = "cuda:" + str(i)
            self.procs.append(PosePredictorMultiGPU.PredictWorker(cfgProc, self.resultQueue, i))

    class PredictWorker(mp.Process):
        def __init__(self, cfg, queue, i):
            self.predictor = None
            self.cfg = cfg
            self.resultQueue = queue
            self.inputs = None
            self.nProc = i
            super().__init__()

        def run(self):
            self.predictor = PosePredictor(self.cfg, self.nProc)
            dataLoader = GazeDataLoader(self.inputs,self.cfg,self.nProc, batchSize= 3 if self.nProc==0 else 4, numWorkers=1)
            for batchData in enumerate(dataLoader):
                outputs = self.predictor.model(batchData[1])
                for output in outputs:
                    out = output['instances']
                    self.resultQueue.append({'gpu': self.nProc, 'boxes': out.pred_boxes.tensor.detach().to("cpu"),
                                             'dpS': out.pred_densepose.S.detach().to("cpu"),
                                             'dpI': out.pred_densepose.I.detach().to("cpu")})
                    del out
                torch.cuda.empty_cache()

    def __call__(self, images):
        t = time.time()
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
        outputsProc2 = []
        for output in self.resultQueue:
            if output['gpu'] == 0:
                outputs.append((output['boxes'], output['dpS'], output['dpI']))
            else:
                outputsProc2.append((output['boxes'], output['dpS'], output['dpI']))
        for output in outputsProc2:
            outputs.append(output)
        print((time.time() - t) / len(images))
        return outputs
