import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

data = open("test.txt", "rt")
lines = data.readlines()

extracted = []
for line in lines:
    splitted = line.split(',')
    extracted.append((float(splitted[0]), float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])))


path = np.zeros(shape = (700, 700, 3), dtype = np.uint8)

for item in extracted:
    x = item[1]
    y = item[2]
    j = int(70*x + 350)
    i = min(max(int(350 - 70*y), 0), 699)
    path[i, j] = (0, 255, 0)

des_path = cv.imread("desired_path.png", cv.IMREAD_COLOR)
cv.imwrite("actual_path.png", cv.add(path, des_path))

time = [x[0] for x in extracted]
eta = [x[3] for x in extracted]
ctrl = [x[4]/500 for x in extracted]

plt.figure()
plt.plot(time, eta)
plt.xlabel("time (s)")
plt.ylabel("eta")
plt.title("eta versus time")
plt.show()

plt.figure()
plt.plot(eta, ctrl)
plt.xlabel("eta")
plt.ylabel("actor output")
plt.title("actor output versus eta")
plt.show()