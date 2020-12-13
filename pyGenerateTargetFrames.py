import cv2
import numpy as np
import pyARAP
import pyTargetImage
import pyTargetFrame
import os

class GenerateTargetFrames:
    def __init__(self, th, tw, no_of_frames, i):
        self.th = th
        self.tw = tw
        self.total_frames = no_of_frames
        self.i = i

    def genFrames(self):
        time = np.array([])

        n = self.i
        frame0 = cv2.imread('OriginalFrames/New%d/frame0.jpg' %n)
        obj0 = pyTargetImage.pyTargetImage(frame0, self.th, self.tw)
        t_frame0 = obj0.target_image()

        directory = "TargetFrames/New%d/" %n
        if not os.path.exists(directory):
            os.makedirs(directory)

        cv2.imwrite('TargetFrames/New%d/t_frame0.jpg' %n , t_frame0)

        for i in range(1,self.total_frames):
            framet0 = cv2.imread('OriginalFrames/New%d/frame%d.jpg' %(n,i-1))
            framet1 = cv2.imread('OriginalFrames/New%d/frame%d.jpg' %(n,i))

            f_obj = pyTargetFrame.pyTargetFrame(framet0, framet1, self.th, self.tw)
            t_frame = f_obj.target_frame()          # generate Target Frames

            t1 = f_obj.calculationTime()            # returns time to calculate vector S for the frame
            time = np.append(time,t1)

            cv2.imwrite('TargetFrames/New%d/t_frame%d.jpg' %(n,i), t_frame)

        return time
