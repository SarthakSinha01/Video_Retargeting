import cv2
import numpy as np
import pyARAP
import pyGridMatching

class temporal_S:
    def __init__(self, frame1, frame2, th, tw):
        self.frame1 = frame1
        self.frame2 = frame2

        self.M = 4
        self.N = 4

        self.th = th
        self.tw = tw


    # this returns the spatial value of S for the frame.
    def prevFrameS(self, frame):
        obj = pyARAP.pyARAP(frame , self.th,self.tw)
        spatial_S = obj.getVector_S()

        return spatial_S


    # function returns the cordinates of region matched with the gridImage.
    def matchedRegionPoints(self, frame1, frame2, i, j):
        obj = pyGridMatching.frame(frame1, frame2)

        grid = obj.getGrid(frame2, i, j)
        H = grid.shape[0]
        W = grid.shape[1]
        cordinates = obj.closestMatchingGrid(frame1, grid)
        cordinates = np.array(cordinates)
        yf = cordinates[0]
        xf = cordinates[1]

        y0,x0 = int(yf/H), int(xf/W)
        y0,x1 = int(yf/H), x0+1
        y1,x0 = y0+1, int(xf/W)
        y1,x1 = y0+1, x0+1

        yd = yf - (y0 * H)
        xd = xf - (x0 * W)
        region_points = np.array([[y0,x0],[y0,x1],[y1,x0],[y1,x1],[yd,xd]])

        return region_points

    # this function returns the value of temporal S vector for the grid in the frame
    def Grid_S_temporal(self, frame1, frame2, i, j):
        vectorS = self.prevFrameS(frame1)
        points = self.matchedRegionPoints(frame1, frame2,i,j)

        y0,x0 = points[0][0], points[0][1]
        y0,x1 = points[1][0], points[1][1]
        y1,x0 = points[2][0], points[2][1]
        y1,x1 = points[3][0], points[3][1]
        yd,xd = points[4][0], points[4][1]


        H0 = frame1.shape[0]
        W0 = frame1.shape[1]

        s_rows = np.array(vectorS[:self.M])
        s_cols = np.array(vectorS[self.M:])

        st_rows = ((1 - (yd*self.M)/H0) * s_rows[y0][0]) + (((yd*self.M)/H0) * s_rows[y1][0])
        st_cols = ((1 - (xd*self.N)/W0) * s_cols[x0][0]) + (((xd*self.N)/W0) * s_cols[x1][0])

        st_grid = np.array([st_rows, st_cols])
        return st_grid

    # this returns the temporal S value for all the grids in the frame.
    def Frame_S_temporal(self, frame1, frame2):
        S_temporal = np.zeros(shape=(self.M, self.N, 2))

        for i in range(0, self.M):
            for j in range(0, self.N):
                st_grid = self.Grid_S_temporal(frame1, frame2, i, j)
                S_temporal[i,j,:] = st_grid

        return S_temporal
