from matplotlib import pyplot as plt

file = open("weights.txt", "rt")
lines = file.readlines()

W = [[], [], [], [], [], [], [], []]

for line in lines:

    splitted = line.split(',')

    for i in range(8):
        W[i].append(float(splitted[i]))

plt.figure()
plt.xlabel("Episode number")
plt.ylabel("Policy network parameters")
plt.title("Policy network parameters versus episode number")

for i in range(8):
    plt.plot(W[i])

plt.show()