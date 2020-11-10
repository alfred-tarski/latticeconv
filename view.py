import numpy as np
import torch
from matplotlib import pyplot as plt

lattice_train_accuracy = torch.load('./lattice_train_accuracy.pt')
lattice_test_accuracy = torch.load('./lattice_test_accuracy.pt')
lattice_train_loss = torch.load('./lattice_train_loss.pt')
conv_train_accuracy = torch.load('./conv_train_accuracy.pt')
conv_test_accuracy = torch.load('./conv_test_accuracy.pt')
# fc_train_accuracy = torch.load('./fc_train_accuracy.pt')
# fc_test_accuracy = torch.load('./fc_test_accuracy.pt')


#n = len(conv_train_accuracy)
n = len(lattice_train_accuracy)

plt.figure(1)
plt.errorbar(1+np.arange(n),lattice_train_accuracy.mean(axis=1),lattice_train_accuracy.std(axis=1), label='Lattice-based CNN')
plt.errorbar(1+np.arange(n),conv_train_accuracy.mean(axis=1), conv_train_accuracy.std(axis=1),label='Classical CNN', alpha=0.5)
#plt.errorbar(1+np.arange(n),fc_train_accuracy.mean(axis=1), fc_train_accuracy.std(axis=1),label='fc', alpha=0.5)

plt.legend()
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.ylim((0, 1))
plt.xlim((0, n+1))
plt.show()


plt.figure(2)
plt.errorbar(1+np.arange(n), lattice_test_accuracy.mean(axis=1), lattice_test_accuracy.std(axis=1), label='Lattice-based CNN')
plt.errorbar(1+np.arange(n), conv_test_accuracy.mean(axis=1), conv_test_accuracy.std(axis=1), label='Classical CNN', alpha=0.5)
#plt.errorbar(1+np.arange(n),fc_test_accuracy.mean(axis=1), fc_test_accuracy.std(axis=1),label='fc', alpha=0.5)

plt.legend()
plt.title('Testing accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.ylim((0, 1))
plt.xlim((0, n+1))
plt.show()

#plt.figure(3)
#for i in range(10):
#    plt.plot(1+np.arange(n), lattice_train_loss[:,i])
#plt.show()