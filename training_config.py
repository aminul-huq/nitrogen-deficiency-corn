import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train(model,train_loader,criterion,optim,epochs,device):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

      #print("Epoch: [{}]  Train_loss: [{:.4f}]".format(epochs+1,train_loss/len(train_set)))
      #print("Epoch: [{}]  Train_loss: [{:.4f}]".format(epochs+1,train_loss))
    return train_loss/len(train_loader), total_correct*100/total

def test(model,test_loader, criterion,optim,filename,epochs,device):
    model.eval()
    test_loss,total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),
                                                                               total_correct*100/total))  
    f = open(filename+".txt","a+")
    f.write('Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}]\n'.format(epochs+1,test_loss/len(test_loader),
                                                                               total_correct*100/total))
    f.close()
      #print("Epoch: [{}]  Test_loss: [{:.4f}]".format(epochs+1,test_loss/len(test_set)))
      #print("Epoch: [{}]  Test_loss: [{:.4f}]".format(epochs+1,test_loss))
    return test_loss/len(test_loader),total_correct*100/total


def test2(model,test_loader, criterion, optim, epochs,device):
    model.eval()
    org_labels, pred_labels  = [],[]

    for i,(images,labels) in enumerate(tqdm(test_loader)):
        org_labels.append(labels.numpy())
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _,predicted = torch.max(outputs.data,1)
        
        pred_labels.append(predicted.cpu().numpy())
    return org_labels, pred_labels
