# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
from typing import List
import torch
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import build_model
import os
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()

class PosePredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.model = build_model(cfg).cuda()
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.outputs = []
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
                    globalOutputs.append(output['instances'])
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
    def __init__(self):
        print("dsfq")

class PosePredictorGroup:
    predictors = dict()

    def __init__(self, cfg, gpus: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        self.predictors = dict()
        self.gpus = gpus
        self.cfg = cfg
        print("init group")
        """print("izi")
        model = build_model(cfg)
        # torch.cuda.set_device(gpu)
        print("izi")
        model.cuda(0)
        print("izi")
        torch.distributed.init_process_group(backend="nccl",rank=0,world_size=2)
        print("izi")
        self.model = DDP(model)
        print("izi")
        self.model.eval()
        print("izi")
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.outputs = []
        assert self.input_format in ["RGB", "BGR"], self.input_format"""

    def predictOutputs(self, inputs: List) -> List:
        inputs1 = [inputs[i] for i in range(0, round(len(inputs) / 2))]
        inputs2 = [inputs[i] for i in range(round(len(inputs) / 2), len(inputs))]
        inputs = [inputs1, inputs2]
        args = {'gpus': self.gpus, 'nodes': 1, 'nr': 0, 'inputs': inputs}
        mp.spawn(self.predictOutputsOneGPU, nprocs=args['gpus'], args=(args,))
        outputs = []
        for predictor in self.predictors:
            for output in predictor.outputs:
                outputs.append(output)
        return outputs

    def predictOutputsOneGPU(self, gpu, args):
        print("init proc", gpu)
        dist.init_process_group(backend='nccl',
                                world_size=self.gpus,
                                rank=gpu)
        print("init model ", gpu)
        torch.manual_seed(42)
        model = PosePredictor(self.cfg.clone(), 0)
        self.predictors["" + str(gpu)] = model
        print(gpu, self, self.predictors)
        print("itération:")
        with torch.no_grad():
            for image in args['inputs'][gpu]:
                if model.input_format == "RGB":
                    image = image[:, :, ::-1]
                height, width = image.shape[:2]
                image = model.transform_gen.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda(0)
                inputs = {"image": image, "height": height, "width": width}
                model.outputs.append(model.model([inputs])[0])
        cleanup()
