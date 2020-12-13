import numpy as np
import math
import cv2

class frame:
    def __init__(self,frame1,frame2):
        self.frame1 = frame1
        self.frame2 = frame2

        self.fr1size = self.frame1.shape
        self.fr2size = self.frame2.shape

        self.W1 = self.fr1size[1]
        self.H1 = self.fr1size[0]

        self.W2 = self.fr1size[1]
        self.H2 = self.fr2size[0]

        self.M = 4
        self.N = 4

    def getGrid(self,frame,i,j):
        H = frame.shape[0]
        W = frame.shape[1]
        h = int(H/self.M)
        w = int(W/self.N)
        y = i*h
        x = j*w
        imgblock = frame[y:y+h, x:x+w]
        return imgblock

    def matchGrid(self,frame,grid):
        gridH = grid.shape[0]
        gridW = grid.shape[1]

        H = frame.shape[0]
        W = frame.shape[1]
        min_cost = math.inf

        hb = int(H - gridH)
        wb = int(W - gridW)

        for i in range(0,hb):
            for j in range(0,wb):
                check_block = frame[i:i+gridH,j:j+gridW]
                dist = np.sqrt(((grid - check_block)**2).sum(axis=1))
                dist = np.mean(dist)
                if dist < min_cost:
                    min_cost = dist
                    x = j
                    y = i
        img = frame[y:y+gridH, x:x+gridW]
        #cv2.imshow("MatchingGrid", img)
        #cv2.waitKey(0)

        return (y,x)

    def getSection(self,frame,grid,yf,xf,h_offset,w_offset):
        H = frame.shape[0]
        W = frame.shape[1]
        gridH = grid.shape[0]
        gridW = grid.shape[1]

        if yf == 0:
            if xf == 0:
                x0 = xf
                x1 = xf + gridW + w_offset
            elif (xf+gridW+w_offset) >= W:
                x0 = xf - w_offset
                x1 = W
            else:
                x0 = xf - w_offset
                x1 = xf + gridW + w_offset
            y0 = yf
            y1 = yf + gridH + h_offset

        elif (yf+ gridH + h_offset) >= H:
            if xf == 0:
                x0 = xf
                x1 = xf + gridW + w_offset
            elif (xf+gridW+w_offset) >= W:
                x0 = xf - w_offset
                x1 = W
            else:
                x0 = xf - w_offset
                x1 = xf + gridW + w_offset
            y0 = yf - h_offset
            y1 = H

        else:
            if xf == 0:
                x0 = xf
                x1 = xf + gridW + w_offset
            elif (xf+ gridW + w_offset) >= W:
                x0 = xf - w_offset
                x1 = W
            else:
                x0 = xf - w_offset
                x1 = xf + gridW + w_offset
            y0 = yf - h_offset
            y1 = yf + gridH + h_offset

        sec = frame[y0:y1, x0:x1]
        anchor_cor = np.array([y0,y1,x0,x1])
        return (anchor_cor)


    def closestMatchingGrid(self, frame, grid):
        gridH = grid.shape[0]
        gridW = grid.shape[1]
        frameH = frame.shape[0]
        frameW = frame.shape[1]
        h = int(frameH/self.M)
        w = int(frameW/self.N)

        min_cost = math.inf

        for i in range(0,self.M):
            for j in range(0,self.N):
                y = i*h
                x = j*w
                block = frame[y:y+h, x:x+w]
                dist = np.sqrt(((grid - block)**2).sum(axis=1))
                dist = np.mean(dist)
                if dist < min_cost:
                    min_cost = dist
                    yf = i*h
                    xf = j*w
                    matching_grid = block

        h_offset = gridH
        w_offset = gridW
        anchor_cor = self.getSection(frame,grid,yf,xf,h_offset,w_offset)
        matching_grid_cor = self.MatchingRegion(frame, anchor_cor, grid,gridH, gridW, yf, xf)

        return(matching_grid_cor)

    def MatchingRegion(self, frame, anchor_cor, grid, Kh, Kw, yf, xf):
        if Kh == 1 or Kw == 1:
            return ([yf,xf])
        else:
            gridH = grid.shape[0]
            gridW = grid.shape[1]

            min_cost = math.inf

            y0 = anchor_cor[0]
            y1 = anchor_cor[1]
            x0 = anchor_cor[2]
            x1 = anchor_cor[3]

            hb = int(y1 - gridH)
            wb = int(x1 - gridW)

            for i in range(y0,hb,int(Kh/2)):
                for j in range(x0,wb,int(Kw/2)):
                    block = frame[i:i+gridH, j:j+gridW]
                    dist = np.sqrt(((grid - block)**2).sum(axis=1))
                    dist = np.mean(dist)
                    if dist < min_cost:
                        min_cost = dist
                        xf = j
                        yf = i
            h_offset = int(Kh/2)
            w_offset = int(Kw/2)
            anchor_cor = self.getSection(frame,grid,yf,xf, h_offset, w_offset)
            #cv2.imshow("Section",section)
            return(self.MatchingRegion(frame, anchor_cor, grid, h_offset,  w_offset,yf,xf))
