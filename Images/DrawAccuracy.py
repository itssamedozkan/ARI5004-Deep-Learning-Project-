import torch

print(torch.cuda)
print(torch.cuda.is_available())

import matplotlib.pyplot as plt

import pandas as pd

accuracy_RMS= []
accuracy_SGD= []
with open("SGD_Accuracy.txt","r") as fp:
    Lines = fp.readlines()
    for line in Lines :
        accuracy_SGD.append(float(line))
        
with open("RMS_Accuracy.txt","r") as fp:
    Lines = fp.readlines()
    for line in Lines :
        accuracy_RMS.append(float(line))

plt.xlabel("Iteration/100 samples")
plt.ylabel("Accuracy")
plt.plot(range(len(accuracy_RMS)), accuracy_RMS ,"-", label = "Training Accuracy with RMSprop")

plt.plot(range(len(accuracy_SGD)), accuracy_SGD ,"g-", label = "Training Accuracy with SGD")
plt.legend()
plt.show()
