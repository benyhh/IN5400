

#from h11 import Data
from importlib import reload
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import sklearn.metrics
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import RainforestDataset
import YourNetwork
reload(RainforestDataset); reload(YourNetwork)
from YourNetwork import SingleNetwork, TwoNetworks
from RainforestDataset import RainforestDataset, ChannelSelect
from torch import Tensor
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import sklearn.metrics

from typing import Callable, Optional

#Seeds
np.random.seed(321)
torch.manual_seed(321)

def train_epoch(model, trainloader, criterion, device, optimizer, network):

    model.train()
    losses = []

    get_rgb = ChannelSelect(channels = [0,1,2])
    get_ir = ChannelSelect(channels = [3])

    for batch_idx, data in enumerate(trainloader):
        #print(data["filename"])
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)

        # TODO calculate the loss from your minibatch.
        inputs = data["image"].to(device)
        labels = data["label"].to(device)
        optimizer.zero_grad()

        if network == "TwoNetworks":
          input1 = inputs[:,:3]
          input2 = inputs[:,3].unsqueeze(1)
          outputs = model(input1, input2)

        else:
          outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        losses.append(loss.item())

        if batch_idx%100==0:
          print('current mean of losses ', np.mean(losses))


        # If you are using the TwoNetworks class you will need to copy the infrared
        # channel before feeding it into your model. 
      
    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl, network):

    model.eval()

    #curcount = 0
    #accuracy = 0 
    
    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)

          fnames.append(data['filename'])
          inputs = data['image'].to(device) 
          labels = data['label'].to(device)

          if network == "TwoNetworks":
            input1 = inputs[:,:3]
            input2 = inputs[:,3].unsqueeze(1)
            outputs = model(input1, input2)

          else:
            outputs = model(inputs)
            
          loss = criterion.forward(outputs, labels)
          losses.append(loss.item())
          
          concat_pred = np.append(concat_pred, outputs.cpu(), axis = 0)
          concat_labels = np.append(concat_labels, labels.cpu(), axis = 0)

          
          # This was an accuracy computation
          # cpuout= outputs.to('cpu')
          # _, preds = torch.max(cpuout, 1)
          # labels = labels.float()
          # corrects = torch.sum(preds == labels.data)
          # accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )
          # curcount+= labels.shape[0]
        
          # TODO: collect scores, labels, filenames
          
    
    for c in range(numcl):
      avgprecs[c] = sklearn.metrics.precision_score(concat_labels[:,c], np.where(nn.Sigmoid()(concat_pred[:,c])>0.5, 1, 0), zero_division=0)
  
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def tailacc(model, dataloader, criterion, device, numcl, network):

    model.load_state_dict(torch.load(f"./models/trained_model_{network}.pt"))
    model.to(device)
    model.eval()
    #curcount = 0
    #accuracy = 0 
    sig = nn.Sigmoid()

    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)

          for fn in data['filename']:
            fnames.append(fn)
          inputs = data['image'].to(device) 
          labels = data['label'].to(device)

          if network == "TwoNetworks":
            input1 = inputs[:,:3]
            input2 = inputs[:,3].unsqueeze(1)
            outputs = model(input1, input2)

          else:
            outputs = model(inputs)
          
          loss = criterion.forward(outputs, labels)
          losses.append(loss.item())
          
          concat_pred = np.append(concat_pred, sig(outputs).cpu(), axis = 0)
          concat_labels = np.append(concat_labels, labels.cpu(), axis = 0)


    for c in range(numcl):
      avgprecs[c] = sklearn.metrics.precision_score(concat_labels[:,c], np.where(concat_pred[:,c]>0.5, 1, 0), zero_division=0)


    concat_labels = concat_labels.T
    concat_pred = concat_pred.T
  
    #Class with highest precision
    prec_max = np.argmax(avgprecs)
    #Sort class with highest precision
    idx_sort = np.argsort(concat_pred[prec_max])
    fnames = np.array(fnames)[idx_sort]
    """
    with open(f"./results/top10img_{network}.txt", 'w+') as f:
      f.write("Top10 Bottom10\n")
      for i in range(10):
        f.write(f"{fnames[i]} {fnames[-1-i]}\n")
    """
    n = 11 #Numer of t-values
    top = 50 #Top x scores
    acc_top = np.zeros((numcl,n))  #Storing precisions 

    t = np.linspace(0.5,np.max(concat_pred[prec_max, idx_sort[-top:]]),n)

    for c in range(numcl):
      for i in range(n):
        c_scores_top = concat_pred[c, idx_sort[-top:]]


        pred_top = np.where(c_scores_top >= t[i], 1, 0)
        if len(pred_top) == 0:
          acc_top[c,i] = np.NaN
        else:
          acc_top[c,i] = sklearn.metrics.accuracy_score(concat_labels[c, idx_sort[-len(pred_top):]], pred_top)

    plt.plot(t,np.mean(acc_top, axis = 0), label = f"Top {top}")
    plt.title(f"Tail accuracy for the top image scores")
    plt.xlabel("t")
    plt.ylabel("Prediction accuracy")
    plt.legend()
    #plt.savefig(f"./plots/tailacc_{network}")
    plt.show()
    plt.clf()

    return 0

    



def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl, network, plot = None):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    
    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer, network)
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()
    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl, network)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      #track current best performance measure and epoch
      best_measure = avgperfmeasure
      best_epoch = epoch
      best_pred = concat_pred
      best_labels = concat_labels
      #save your scores
      
  write_data(network, [best_epoch, best_measure, best_pred])
  torch.save(bestweights, f"./models/trained_model_{network}.pt")
  


  if plot:
    plt.plot(range(num_epochs), trainlosses, label = "Train")
    plt.plot(range(num_epochs), testlosses, label = "Validation")
    plt.legend()
    plt.title("Performance curves")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()
    #plt.savefig(f"./plots/Perfcurves_{network}")
    plt.clf()
  

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs


class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(nn.modules.loss._Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction = reduction)
        
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return self.criterion(input = input_, target = target.float())




def runstuff(network="SingleNetwork", pretrained = False, reproduction = False):
  config = dict()
  config['use_gpu'] = True #True
  config['lr'] = 0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17 

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]), 
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'train_rgb': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect(channels=[0,1,2]),
        transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
    ]),
      'val_rgb': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelSelect(channels=[0,1,2]),
        transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137])
    ]),
      'train_all': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #ChannelSelect(channels=[3]),
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ]),
      'val_all': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #ChannelSelect(channels=[3]),
        transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
    ])
  }

  root_dir = "/itf-fi-ml/shared/IN5400/2022_mandatory1/"

  # Datasets
  image_datasets={}
  image_datasets['train_rgb'] = RainforestDataset(root_dir=root_dir,trvaltest=0, transform=data_transforms['train_rgb'])
  image_datasets['val_rgb'] = RainforestDataset(root_dir=root_dir,trvaltest=1, transform=data_transforms['val_rgb'])
  image_datasets['train_all'] = RainforestDataset(root_dir=root_dir,trvaltest=0, transform=data_transforms['train_all'])
  image_datasets['val_all'] = RainforestDataset(root_dir=root_dir,trvaltest=1, transform=data_transforms['val_all'])
  # Dataloaders
  num_workers=4
  dataloaders = {}
  dataloaders['train_rgb'] = DataLoader(image_datasets["train_rgb"], batch_size=config['batchsize_train'], shuffle=True, num_workers = num_workers)
  dataloaders['val_rgb'] = DataLoader(image_datasets["val_rgb"], batch_size=config['batchsize_val'], shuffle=False, num_workers = num_workers)
  dataloaders['train_all'] = DataLoader(image_datasets["train_all"], batch_size=config['batchsize_train'], shuffle=True, num_workers = num_workers)
  dataloaders['val_all'] = DataLoader(image_datasets["val_all"], batch_size=config['batchsize_val'], shuffle=False, num_workers = num_workers)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')



  # Model
  #create an instance of the network that you want to use.
  if network == "TwoNetworks":
    model = TwoNetworks()
    model = model.to(device)
    trainloader = dataloaders["train_all"]
    valloader = dataloaders["val_all"]

  elif network == "SingleNetwork_all":
    model = SingleNetwork(weight_init = "kaiminghe").net
    model = model.to(device)
    trainloader = dataloaders["train_all"]
    valloader = dataloaders["val_all"]
  else:
    model = SingleNetwork().net
    model = model.to(device)
    trainloader = dataloaders["train_rgb"]
    valloader = dataloaders["val_rgb"]
  

  lossfct = yourloss()
  someoptimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
  
  # Decay LR by a factor of 0.3 every X epochs
  # Observe that all parameters are being optimized
  somelr_scheduler = lr_scheduler.StepLR(optimizer=someoptimizer, step_size=config["scheduler_stepsize"], gamma=config['scheduler_factor'])


  if pretrained is not True:
    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(trainloader, valloader,
                        model, lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'], network = network)
    write_data(network, [best_epoch, best_measure, trainlosses, testlosses, testperfs])
    if network == "SingleNetwork":
      tailacc(model, valloader , lossfct, device, config["numcl"], network = network)

  else:
    model.load_state_dict(torch.load(f"./models/trained_model_{network}.pt"))
    model.to(device)
    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, valloader, lossfct, device, config['numcl'], network)
    

  if reproduction:
    best_epoch, best_measure_saved, pred_scores_saved = read_data(network)
    RE = np.mean((pred_scores_saved-concat_pred)/concat_pred)
    print("Network:", network.replace("_all"," 4 channels"))
    print("Mean relative error of saved and reproduced prediction scores: ", RE)


def write_data(network,data):
  fn = network + f"_scores.txt"
  preds = data[2]
  with open("./results/" + fn,'w+') as f:
        f.write("Best epoch\n")
        f.write(str(data[0]))
        f.write("\nBest measure\n")
        f.write(str(data[1]))
        f.write("\n\nClass predictions scores\n")
        
        for i in preds:
          for j in i:
            f.write(f"{j} ")
          f.write("\n")

def read_data(network):
  fn = network + "_scores.txt"

  with open("./results/" + fn, 'r') as f:
    f.readline()
    best_epoch = int(f.readline())
    f.readline()
    best_measure = float(f.readline())
    f.readline()
    f.readline()

    pred_scores = np.empty((0, 17))

    for line in f:

      split = np.array(line.split(), dtype = float) #np.array(line.split(), dtype = float)
      pred_scores = np.append(pred_scores, split.reshape(1,17), axis = 0)


  return best_epoch, best_measure, pred_scores



if __name__=='__main__':
    #Reproduction routine, compares saved models validation scores from the ones in txt file
    runstuff("SingleNetwork", pretrained = True, reproduction = True)
    runstuff("SingleNetwork_all", pretrained = True, reproduction = True)
    runstuff("TwoNetworks", pretrained = True, reproduction = True)

    # Uncomment to see models training
    #runstuff("SingleNetwork")
    #runstuff("SingleNetwork_all")
    #runstuff("TwoNetworks")

