import time

import torch
from gaze360.code.model import GazeLSTM

class GazePredictor:
    def __init__(self, gazeModelFile):
        self.model = torch.nn.DataParallel(GazeLSTM()).cuda()
        STATE_DICT = torch.load(gazeModelFile)['state_dict']
        self.model.load_state_dict(STATE_DICT)
        self.model.eval()

    def __call__(self, input):
        return self.model(input)

    def predictMultipleInputs(self, batchedInputs):
        numberImages = len(batchedInputs)
        batchSize = 96
        batchNumber = int(round(numberImages / batchSize))
        if batchNumber < numberImages / batchSize:
            batchNumber += 1
        underTotal = 0
        batches = []
        for j in range(batchNumber):
            batches.append(
                torch.cat([*batchedInputs[j * batchSize:j * batchSize + min(batchSize, numberImages - underTotal)]],
                          dim=0))
            underTotal += batchSize
        outputs = []
        for i in range(batchNumber):
            out = self.model(batches[i])[0]
            for o in out:
                outputs.append(o.detach())
            del out
            torch.cuda.empty_cache()
        return outputs


