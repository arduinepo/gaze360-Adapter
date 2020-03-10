from math import floor, ceil
import torch
from torch.utils.data import Dataset, DataLoader
import detectron2.data.transforms as T
import torchvision.transforms as transformsPIL
import transforms as transformsCV

IMAGE_NORMALIZE = transformsCV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class GazeDataSet(Dataset):
    def __init__(self, frames, instances, W):
        self.W = W
        self.inputs = []
        self.instances = instances
        self.frames = frames
        print("input transformation for gaze inference")
        for i in range(min(instances.keys()), max(instances.keys()) + 1):
            if i % 99 == 0:
                print("frame", i)
            if i in instances:
                for id_t in instances[i].keys():
                    self.inputs.append(self.getInputImage(i, id_t))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item]

    def getInputImage(self, indexImage, indexInstance):
        input_image = torch.zeros(7, 3, 224, 224)
        count = 0
        for j in range(indexImage - 3 * self.W, indexImage + 4 * self.W, self.W):
            if j in self.instances and indexInstance in self.instances[j]:
                new_im = self.frames[j]
                bbox, eyes = self.instances[j][indexInstance]
            else:
                new_im = self.frames[indexImage]
                bbox, eyes = self.instances[indexImage][indexInstance]
            x, y, w, h = int(floor(bbox[0])), int(floor(bbox[1])), int(ceil(bbox[2])), int(ceil(bbox[3]))
            im = transformsCV.Resize((224, 224))(new_im[y:h, x:w])
            input_image[count, :, :, :] = IMAGE_NORMALIZE(transformsCV.ToTensor()(im))
            count += 1
        return input_image.view(1, 7, 3, 224, 224)

class GazeDataLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset, batch_size=96, shuffle=False, num_workers=1)

class DensePoseDataSet(Dataset):
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

class DensePoseDataLoader(DataLoader):
    def __init__(self, inputs, cfg, gpu, batchSize):
        super().__init__(DensePoseDataSet(inputs, cfg, gpu), batch_size=batchSize, shuffle=False, num_workers=1,
                         collate_fn=collate)

def collate(batch):
    return [img for img in batch]
