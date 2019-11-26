import numpy as np

training_number = 100000
testing_number = 1000
length = 100
char = 26
X = []
y = []
s = 0

for i in range(training_number + testing_number):
    t = np.random.randint(1, length + 1)
    x = np.zeros(length)
    cnt_a = 0
    for j in range(t):
        x[j] = np.random.randint(1, char + 2)
        if x[j] == 1:
            cnt_a += 1
    X.append(x)
    if cnt_a > 2:
        y.append(1)
        s += 1
    else:
        y.append(0)

print(s / (training_number + testing_number))
np.save("X_test", X[-testing_number:])
np.save("y_test", y[-testing_number:])
np.save("X_train", X[:training_number])
np.save("y_train", y[:training_number])
