import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

def train(model, criterion, optimizer, train_data, test_data, n_epochs, device, callback=None):
    train_accuracy = torch.zeros(n_epochs)
    test_accuracy = torch.zeros(n_epochs)
    train_loss = torch.zeros(n_epochs)

    for epoch in range(n_epochs): 
        total = 0.0
        total_loss = 0.0
        correct = 0
        model.train()
        for i, data in enumerate(train_data):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
        train_accuracy[epoch] = correct/total
        train_loss[epoch] = total_loss/total
        #print("Training took {:.1f} seconds".format(time.time() - start_time))
        #print("Training accuracy: {:.1%}".format(train_accuracy[epoch,trial]))
        #print('Testing accuracy of LatticeClassifier...')
        total = 0.0
        correct = 0
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(test_data):
                inputs,labels = data[0].to(device),data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy[epoch] = correct/total
        if(epoch % 10 == 9):
            print("Epoch {:d}".format(epoch+1))
            print("Training accuracy: {:.1%}".format(train_accuracy[epoch]))
            print("Testing accuracy: {:.1%}".format(test_accuracy[epoch]))
            if(callback is not None):
                callback(model,epoch)
    return train_accuracy, test_accuracy, train_loss