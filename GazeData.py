from torch.utils.data import Dataset

class GazeDataSet(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        return self.frames[item]
