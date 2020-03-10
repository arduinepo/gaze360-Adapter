import torch
from gaze360.code.model import GazeLSTM

class GazePredictor:
    def __init__(self, gazeModelFile):
        self.model = torch.nn.DataParallel(GazeLSTM()).cuda()
        STATE_DICT = torch.load(gazeModelFile)['state_dict']
        self.model.load_state_dict(STATE_DICT)
        self.model.eval()

    def __call__(self, inputs):
        return self.model(inputs[1])[0]
