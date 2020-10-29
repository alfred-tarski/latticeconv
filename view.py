import numpy as np
import torch
from matplotlib import pyplot as plt

lattice_train_accuracy = torch.load('./lattice_train_accuracy.pt')
lattice_test_accuracy = torch.load('./lattice_test_accuracy.pt')
conv_train_accuracy = torch.load('./conv_train_accuracy.pt')
conv_test_accuracy = torch.load('./conv_test_accuracy.pt')
fc_train_accuracy = torch.load('./fc_train_accuracy.pt')
fc_test_accuracy = torch.load('./fc_test_accuracy.pt')


n = len(lattice_train_accuracy)

plt.errorbar(1+np.arange(n),lattice_train_accuracy.mean(axis=1),lattice_train_accuracy.std(axis=1), label='lattice')
plt.errorbar(1+np.arange(n),conv_train_accuracy.mean(axis=1), conv_train_accuracy.std(axis=1),label='conv2d', alpha=0.5)
plt.errorbar(1+np.arange(n),fc_train_accuracy.mean(axis=1), fc_train_accuracy.std(axis=1),label='fc', alpha=0.5)

plt.legend()
plt.title('train')
plt.ylim((0, 1))
plt.xlim((0, n+1))
plt.show()


plt.errorbar(1+np.arange(n), lattice_test_accuracy.mean(axis=1), lattice_test_accuracy.std(axis=1), label='lattice')
plt.errorbar(1+np.arange(n), conv_test_accuracy.mean(axis=1), conv_test_accuracy.std(axis=1), label='conv2d', alpha=0.5)
plt.errorbar(1+np.arange(n),fc_test_accuracy.mean(axis=1), fc_test_accuracy.std(axis=1),label='fc', alpha=0.5)

plt.legend()
plt.title('test')
plt.ylim((0, 1))
plt.xlim((0, n+1))
plt.show()