import torch
from torch.utils.data import Dataset, DataLoader
import detectron2.data.transforms as T

class GazeDataSet(Dataset):
    def __init__(self, frames, cfg, gpu):
        self.frames = frames
        self.gpu = gpu
        self.input_format = cfg.INPUT.FORMAT
        self.transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                                                  cfg.INPUT.MAX_SIZE_TEST)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        img = self.frames[item]
        if self.input_format == "RGB":
            img = img[:, :, ::-1]
        height, width = img.shape[:2]
        return {"image": self.transform(img), "height": height, "width": width}

    def transform(self, image):
        return torch.as_tensor(self.transform_gen.get_transform(image)
                               .apply_image(image).astype("float32").transpose(2, 0, 1)).cuda(self.gpu)

class GazeDataLoader(DataLoader):
    def __init__(self,inputs,cfg,numberGpus,batchSize,numWorkers):
        super().__init__(GazeDataSet(inputs,cfg,numberGpus),batch_size=batchSize,shuffle=False,num_workers=numWorkers,collate_fn=collate)

def collate(batch):
    return [img for img in batch]
