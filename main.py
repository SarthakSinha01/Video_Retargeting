import numpy as np
import cv2
import pyVideoToFrames
import time
import os


th = 0
tw = 80

tw_arr = np.array([40,80,50,40,80,70,70,40,40,40])      # this array contains the number of columns to be removed from ith test video

for i in range(1,11):
    tw = tw_arr[i-1]

    time0 = time.process_time()
    video = 'OriginalVideo/New/test%d.mp4'%i

    vidObj = pyVideoToFrames.inputOutputVideo(video,th,tw,i)
    count = vidObj.generateFrames()           # generateFrames() generate origianl frames and return the total no of frames in the video

    time1 = time.process_time()

    SforFrames = vidObj.genTargetFrames() # genTargetFrames() generate target frames and return time to compute S for each frame

    time2 = time.process_time()
    vidObj.generateVideo()              # generateVideo() generate target video from target frames
    time3 = time.process_time()

    directory = "Time/Time"

    file = open("Time/TimeFile%d.txt" %i,"w")
    file.write("Test Video :: test%d\n"%i)
    file.write("*******************************************\n")
    file.write("Total Frames :: %d\n"%(count-1))
    file.write("------------------------------------------------------------------------------\n")
    file.write("Process Description                                  Time (in seconds)\n")
    file.write("------------------------------------------------------------------------------\n")
    file.write("1. Video to Frames conversion time                ::  %s\n"%(time1-time0))
    file.write("2. Time to generate target frames                 ::  %s\n"%(time2-time1))
    file.write("3. TargetFrames to TargetVideo conversion time    ::  %s\n"%(time3-time2))
    file.write("4. Total running time (I/P Video to O/P Video)    ::  %s\n"%(time3-time0))
    file.write("**************************************************************************\n")
    file.write("Time to calculate vector S for each frame (starting from second frame)\n")
    file.write("--------------------------------------------------------------------------\n")
    for i in (SforFrames):
        file.write("%f,"%i)
    file.write("\n")
    file.close()

    print("DONE")
