import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math

read = pandas.read_csv('RBE500-F17-100ms-Constant-Vel.csv', header=None)
data = read.astype(float)
data = np.array(data)

time = []
for i in range(0,len(data)):
    time.append(i/10)

New_measurement = []
std_x = []
std_v = []
corr = []

X = np.matrix('2530;-10')
A = np.matrix('1, 0.1; 0, 1')
p_noise = np.matrix('0.1,1;1,10')
P = np.matrix('100,0;0,100')
q = 10
H = np.matrix('1,0')
I = np.matrix('1,0;0,1')

for i in data:
    x_kp = A * X
    p_kp = A * P * (A.transpose()) + p_noise
    num = p_kp * H.transpose()
    den = H * p_kp * H.transpose() + q
    K_gain = num / den
    X = x_kp + K_gain * (i - (H * x_kp))
    P = (I - (K_gain*H)) * p_kp
    std_x.append(math.sqrt(P[0,0]))
    std_v.append(math.sqrt(P[1,1]))
    corr.append(P[0,1] / ((math.sqrt(P[0,0])) * (math.sqrt(P[1,1]))))
    New_measurement.append(X)

New_measurement=np.array(New_measurement)
plt.plot(time,data,label = 'Raw Data')
plt.plot(time, New_measurement[:,0],label = 'New measurement')
plt.legend(loc = 'upper right')
plt.xlabel('Time in seconds')
plt.ylabel('Raw data and Estimated Position in cm')
plt.title('Question A - Position Data')

plt.show()

plt.plot(time,New_measurement[:,1])
plt.xlabel('Time in seconds')
plt.ylabel('Velocity in cm/s')
plt.title('Question A - Velocity Data')
plt.show()

plt.plot(time,std_x,label = 'Std_Position')
plt.plot(time,std_v,label = 'Std_Velocity')
plt.legend(loc = 'upper right')
plt.xlabel('Time in seconds')
plt.ylabel('Standard Deviation in cm')
plt.title('Question A - Standard Deviation of Position and Velocity')
plt.show()

plt.plot(time,corr)
plt.xlabel('Time in seconds')
plt.ylabel('Correlation Coefficient')
plt.title('Question A - Correlation Coefficient')
plt.show()
