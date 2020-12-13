import cv2
import pyARAP
import pyTotalARAP
import time
import numpy as np

class pyTargetFrame:
    def __init__(self,image,tgFrame, th,tw):
        self.image = image
        self.tgFrame = tgFrame
        self.imagesize = self.image.shape
        self.W = self.imagesize[1]
        self.H = self.imagesize[0]
        self.M = 4
        self.N = 4
        self.th = th
        self.tw = tw

        #self.start = time.process_time()

        self.obj = pyTotalARAP.pyTotalARAP(self.image,self.tgFrame, self.th, self.tw)
        self.time,self.STotal = self.obj.getVector_Stotal()

        #self.end = time.process_time()
        #self.s_rows = np.array(self.vector[:self.M])
        #self.s_cols = np.array(self.vector[self.M:])
        #print(self.s_rows)
        #print('COL VECTOR')
        #print(self.s_cols)

    def calculationTime(self):
        #time = self.end - self.start
        return self.time

    # gridImage extracts the grid section of the image or frame and returns it.

    def gridImage(self,i,j):
        h = int(self.H/self.M)
        w = int(self.W/self.N)
        y = i*h
        x = j*w
        img_section = self.image[y:y+h, x:x+w]
        return img_section

    # this returns the target section of the required gridImage of a frame
    def target_gridImage(self, i,j):
        h = int(self.H/self.M)  # old height
        w = int(self.W/self.N)  # old width
        y = i*h
        x = j*w
        img_section = self.image[y:y+h, x:x+w]


        tg_ht = self.STotal[i][0]
        tg_wd = self.STotal[self.M+j][0]
        #print(tg_ht,tg_wd)

        ar = tg_wd/(self.W/self.N)  # ar is aspect ratio
        dim = (int(tg_wd), int(tg_ht))
        target_section = cv2.resize(img_section, dim, interpolation=cv2.INTER_AREA)
        #print(target_section.shape)
        return target_section


    # target_frame combines all the target section to produce the target frame
    def target_frame(self):
        for i in range(0, self.M):
            r_part = self.target_gridImage(i,0)
            for j in range(1, self.N):
                c_sec = self.target_gridImage(i,j)
                r_part = np.concatenate((r_part,c_sec),axis=1)

            if(i == 0):
                rows = r_part
            else:
                rows = np.concatenate((rows,r_part),axis=0)
        t_img = rows

        #cv2.imshow("TImg", t_img)
        #cv2.waitKey(0)
        return t_img
