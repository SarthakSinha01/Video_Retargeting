import cv2
import numpy
import pyGenerateTargetFrames
import os

class inputOutputVideo:
    def __init__(self, video, th, tw, i):
        self.video = video
        self.th = th
        self.tw = tw
        self.i = i
        #self.fps = fps

        self.vidObj = cv2.VideoCapture(self.video)
        self.fps = self.vidObj.get(cv2.CAP_PROP_FPS)


    def generateFrames(self):
        vidObj = cv2.VideoCapture(self.video)
        count = 0
        success = 1
        i = self.i

        directory = "OriginalFrames/New%d/" %i
        if not os.path.exists(directory):
            os.makedirs(directory)

        while success:
            success,image = vidObj.read()
            cv2.imwrite("OriginalFrames/New%d/frame%d.jpg"%(i,count), image)
            count += 1

        return count

    def genTargetFrames(self):
        i = self.i
        N = self.generateFrames()
        obj = pyGenerateTargetFrames.GenerateTargetFrames(self.th, self.tw, N-1, i)

        time = obj.genFrames()
        return time

    def generateVideo(self):
        n = self.i
        img = cv2.imread('TargetFrames/New%d/t_frame1.jpg' %n)
        H = img.shape[0]
        W = img.shape[1]
        fps = self.fps

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("TargetVideo/target_video%d.avi" %n, fourcc, fps, (W,H))

        total_frames = self.generateFrames()

        for i in range(1,total_frames):
            frame = cv2.imread('TargetFrames/New%d/t_frame%d.jpg' %(n,i))
            #print(frame.shape)
            out.write(frame)
