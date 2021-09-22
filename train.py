# [1]
import os
path = os.getcwd()
data_dir = os.path.join(path + os.sep, 'data')

import numpy as np
import sys
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset

# [2]
def extract_path(puth):
    return puth[0]

# [3]
train = datasets.ImageFolder(os.path.join(data_dir + os.sep, "train"))
validation = datasets.ImageFolder(os.path.join(data_dir + os.sep, "validation"))

train_imgs = pd.Series(train.imgs, name = "path").apply(func=extract_path)
validation_imgs = pd.Series(validation.imgs, name = "path").apply(func=extract_path)

train_target = pd.Series(train.targets, name ="label")
validation_target = pd.Series(validation.targets, name ="label")

# [4]
train = pd.concat([train_imgs, train_target],axis=1)
val = pd.concat([validation_imgs, validation_target],axis=1)

# [5]
from torch.utils.data import Dataset
from PIL import Image

class DataPrepararion(Dataset):
    
    def __init__(self, data, transform_pipe, device):
        self.data = data
        self.transform_pipe = transform_pipe
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx, :]
        image = Image.open(item["path"]).convert("RGB")
        
        image = self.transform_pipe(image).to(self.device)
        if item["label"] == 0:
            label = torch.zeros(size=(1,1)).to(self.device)
        else:
            label = torch.ones(size=(1,1)).to(self.device)
        return [image, label]

# [6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# [7]
from torchvision import transforms

transform_pipe = transforms.Compose([transforms.Resize((150,150),interpolation=Image.BILINEAR),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

train_dataset = DataPrepararion(train, transform_pipe, device)
val_dataset = DataPrepararion(val, transform_pipe, device)

# [8]
from torch.utils.data import DataLoader, RandomSampler

batch_size = 32

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          sampler=RandomSampler(data_source=train_dataset))
val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        sampler=RandomSampler(data_source=val_dataset))

# [9]
model = models.vgg16(pretrained=True)
print(model)

# [10]
for param in model.parameters():
    param.requires_grad = False

# [11]
from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(25088,100)),
    ('relu1', nn.ReLU()),
    ('dropout1',nn.Dropout(0.3)),
    ('fc2', nn.Linear(100,1)),
    ('output', nn.Sigmoid())
]))

model.classifier = classifier

# [12]
learning_rate = 0.001
# Optimizer
optimizer = optim.RMSprop(model.classifier.parameters(),lr =0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.BCELoss() #cross entropy loss

# [13]
model = model.to(device)

# [14]
def get_class_accuracy(arr):
    # first column: true value, second column: predicted value
    # find number of missclassifications for class = 0
    true_0_all = arr[arr[:,0]==0]
    class_0_acc = (true_0_all[:,1]==0).sum() / true_0_all.shape[0]
    true_1_all = arr[arr[:,0]==1]
    class_1_acc = (true_1_all[:,1]==1).sum() / true_1_all.shape[0]
    return class_0_acc, class_1_acc

# [15]
n_epochs = 10

# [16]
from sklearn.metrics import accuracy_score
accuracy_stats = {
    'train': [],
    "val": []
}

all_preds = np.empty((1, 2))

for epoch in range(n_epochs):
    print("=== Epoch", epoch+1, "===")
    acc_train = 0
    acc_val = 0
    # train
    model.train()  # train mode
    for x, y in train_loader:
        y = y.view(-1, 1)
        
        optimizer.zero_grad() 
        probas = model(x) 
        loss = loss_func(probas, y) 
        loss.backward()
        optimizer.step()
        
    model.eval()  # evaluation mode

    for x, y in train_loader:
        y = y.view(-1, 1)
        with torch.no_grad():
            probas = model(x)
        pred = np.round(probas.cpu().numpy())
        acc_train += accuracy_score(y_true=y.cpu().numpy(), y_pred=pred)
    
    acc_train /= len(train_loader)
    accuracy_stats['train'].append(acc_train)
    print("Train Accuracy:", acc_train)
    
    for x, y in val_loader:
        y = y.view(-1, 1)
        with torch.no_grad():
            probas = model(x)
        pred = np.round(probas.cpu().numpy())
        acc_val += accuracy_score(y_true=y.cpu().numpy(), y_pred=pred)
        true_pred = np.hstack((y.cpu().numpy(), pred))
        all_preds = np.vstack((all_preds, true_pred))
        
    acc_val /= len(val_loader)
    accuracy_stats['val'].append(acc_val)
    print("Val Accuracy:", acc_val, "\n")

cat_acc, dog_acc = get_class_accuracy(all_preds[1:])
df = pd.DataFrame({
    'Train_accuracy': [accuracy_stats['train'][-1]],
    'Val_accuracy': [accuracy_stats['val'][-1]],
    'Cat_accuracy' : [cat_acc],
    'Dog_accuracy' : [dog_acc]
             })

csvlogger = os.path.join(os.getcwd() + os.sep, 'metrics.csv')
df.to_csv(csvlogger, index=False)

# [17]
checkpoints = {
     'pre-trained':'vgg16',
     'classifier':nn.Sequential(OrderedDict([
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(25088,100)),
    ('relu1', nn.ReLU()),
    ('dropout1',nn.Dropout(0.3)),
    ('fc2', nn.Linear(100,1)),
    ('output', nn.Sigmoid())
])),
    'state_dict':model.state_dict()
}

top_model_weights_path = os.path.join(os.getcwd() + os.sep, 'model.pth')
torch.save(checkpoints,top_model_weights_path)