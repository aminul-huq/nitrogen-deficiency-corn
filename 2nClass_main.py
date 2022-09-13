import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,random_split,Dataset
import torch.optim as optim
from networks import *
from training_config import *
import matplotlib.pyplot as plt
import seaborn as sns
import random,argparse, pickle
torch.manual_seed(0)
random.seed(0)
print("2 N classification")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Corn')

    parser.add_argument('--epochs', default=100, metavar='epochs', type=int, help='Number of epochs')
    parser.add_argument('--net', default=1, metavar='net', type=int, help='0 VGG16, 1 AlexNet')
    parser.add_argument('--fname', default='filename', metavar='fname', type=str, help='file name')
    
    args = parser.parse_args()
    EPOCHS = args.epochs
    NET = args.net    
    FN = args.fname
    
    
with open('/home/aminul/unr/dataset/224_data.pkl', "rb") as fp:   # Unpickling
    x,y1,y2,y3 = pickle.load(fp)
print("Dataset loaded")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    ]) 

class NewDataset(Dataset):
    def __init__(self,data,labels1,labels2,labels3,transform=None):
        self.data = data
        self.label1 = labels1
        self.label2 = labels2
        self.label3 = labels3
        self.transform = transform
    def __len__(self):
        return len(self.data)    
    def __getitem__(self,idx):
        image = self.data[idx]
        label1 = self.label1[idx]
        label2 = self.label2[idx]
        label3 = self.label3[idx]
        return self.transform(image), label1, label2, label3



new_trainset = NewDataset(x,y1,y2,y3,transform_train)
lengths = [int(len(new_trainset)*0.8), int(len(new_trainset)*0.2)]
trainset,testset = random_split(new_trainset,lengths)


train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
test_loader = DataLoader(testset, batch_size=10, shuffle=False)


device = torch.device('cuda')
 
if NET == 0:
    print("VGG16")
    net = VGG('VGG16', num_classes = 2).to(device)
else:
    print("AlexNet")
    net = AlexNet(num_classes = 2).to(device)    
    
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr=0.01)


def train(model,train_loader,criterion,optim,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images, labels,_,_) in enumerate(tqdm(train_loader)):
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

    return train_loss/len(train_loader), total_correct*100/total

def test(model,test_loader, criterion,optim, filename, epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0

    for i,(images, labels,_,_) in enumerate(tqdm(test_loader)):
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
    
    return test_loss/len(test_loader),total_correct*100/total


n_epochs = EPOCHS
filename = FN
train_loss,test_loss, train_acc, test_acc = [],[],[],[]

for i in range(n_epochs):
    a,b = train(net, train_loader, criterion, optim, i)
    c,d = test(net, test_loader, criterion, optim, filename, i)
    train_loss.append(a), test_loss.append(c),train_acc.append(b), test_acc.append(d)


fig = plt.figure(figsize=(12,4))
sns.set_theme()
fig.tight_layout() 

plt.subplot(1, 2, 1)
plt.plot(train_loss,label='train_loss')
plt.plot(test_loss,label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc,label='train_acc')
plt.plot(test_acc,label='test_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.suptitle("VGG16-2-N-Class Classification")
plt.savefig('/home/aminul/unr/imgs/vgg_2nclass.png',dpi=300)



























