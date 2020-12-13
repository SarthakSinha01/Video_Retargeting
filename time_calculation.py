import numpy as np
import matplotlib.pyplot as plt

time = np.zeros(shape = (10,60))
time_perFrame = np.zeros(shape = (10,60))

for i in range(0,10):
    file = open("Time/TimeFile%d.txt"%(i+1),"r")

    a = file.read().split(',')
    a = np.array(a)
    len = a.shape[0]-1

    if len > 60:
        len = 60

    for j in range(len):
        time[i][j] = float(a[j])

    file.close()


avg_time = np.mean(time, axis=0)
file = open("Time(Average).txt","w")
    for i
file.write(str(avg_time))
file.close()
#print(avg_time)

x = np.zeros(shape=(60,))
for i in range(60):
    x[i] = i+1

y5 = time[2]
y6 = time[4]
y = avg_time


fig = plt.figure()
plt.xlabel("Frame No")
plt.ylabel("Time (seconds)")
plt.title("Time for each frame")
plt.plot(x,y5,color='g', figure = fig)
plt.savefig("Fig5(Horse)")

plt.plot(x,y6,color='r',figure = fig)
plt.savefig("Fig6(Match)")

plt.plot(x,y,color='b', figure = fig)
plt.savefig("Average time for all videos")
