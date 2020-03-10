from GazeExtractor import GazeExtractor
from GazeVisualizer import GazeVisualizer
import matplotlib
import sys,time
sys.path.append("/home/pandrieu/dev/tlab/DensePose/")
matplotlib.use('TkAgg')
dir = "/home/pandrieu/dev/tlab/"

def main():
    begin,nFrames = 430,1000
    WIDTH, HEIGHT = 960,720
    configArgs = {'config': dir + "DensePose/configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml",
                  'poseModel': dir + "DensePose/R-101.pkl",
                  'gazeModel': dir + "gaze360/gaze360_model.pth.tar"}
    paths = {'in': "/home/pandrieu/dev/tlab/videos/g36sample.mp4", 'out': dir + "out1.mp4"}
    paths = {'in': "/home/pandrieu/Public/downldVideosTeaching/web/12th Grade ELA.mp4", 'out': dir + "out2.mp4"}
    paths1={'in': "/home/pandrieu/dev/tlab/video.mp4", 'out': dir + "out1.mp4"}
    paths1 = {'in': "/home/pandrieu/Public/downldVideosTeaching/web/tri/frontStudents/1.mp4", 'out': dir + "out1.mp4"}
    ext = GazeExtractor(configArgs['config'], configArgs['poseModel'], configArgs['gazeModel'], WIDTH, HEIGHT,0.80,2)
    viz = GazeVisualizer(ext)
    t=time.time()
    viz.generateGazeVideo(paths['in'], paths['out'], begin, nFrames)
    print((time.time()-t)/nFrames)

if __name__ == "__main__":
    main()
