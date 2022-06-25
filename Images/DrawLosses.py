import torch

print(torch.cuda)
print(torch.cuda.is_available())

import matplotlib.pyplot as plt

import pandas as pd


obj2 = pd.read_pickle(r'train_accuracy_rms')
obj3 = pd.read_pickle(r'train_accuracy_SGD')



plt.plot(range(len(obj2)), obj2 ,"-", label = "Training Accuracy with RMSprop")

plt.plot(range(len(obj3)), obj3 ,"g-", label = "Training Accuracy with SGD")
plt.legend()
plt.show()
