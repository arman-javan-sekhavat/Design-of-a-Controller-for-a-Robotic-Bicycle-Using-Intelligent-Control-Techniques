from matplotlib import pyplot as plt

file = open("PID.txt", "rt")
lines = file.readlines()

P = []
I = []
D = []

for line in lines[0:100]:

    splitted = line.split(',')
    P.append(float(splitted[0]))
    I.append(float(splitted[1]))
    D.append(float(splitted[2]))

plt.figure()
plt.xlabel("Episode number")
plt.ylabel("PID parameters")
plt.title("PID parameters versus episode number")
plt.plot(P, label = "Kp")
plt.plot(I, label = "Ki")
plt.plot(D, label = "Kd")
plt.legend()
plt.show()