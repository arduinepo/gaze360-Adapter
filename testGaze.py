from GazeExtractor import GazeExtractor
from GazeVisualizer import GazeVisualizer
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
matplotlib.use('TkAgg')

dir = "/home/pandrieu/dev/tlab/"

def main():
    WIDTH, HEIGHT = 1920, 1440
    configArgs = {'config': dir + "DensePose/configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml",
                  'poseModel': dir + "DensePose/R-101.pkl",
                  'gazeModel': dir + "gaze360/gaze360_model.pth.tar"}
    paths1 = {'in': "/home/pandrieu/dev/tlab/videos/g36sample.mp4", 'out': dir + "out1.mp4"}
    paths = {'in': "/home/pandrieu/dev/tlab/video.mp4", 'out': dir + "out1.mp4"}
    ext = GazeExtractor(configArgs['config'], configArgs['poseModel'], configArgs['gazeModel'], WIDTH, HEIGHT)
    viz = GazeVisualizer(ext)
    viz.generateGazeVideoFromInput(paths['in'], paths['out'], 0, 10)

if __name__ == "__main__":
    main()
