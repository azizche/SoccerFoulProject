import json
from pathlib import Path
from torchvision.io import read_video
import numpy as np
import torch
from SoccerFoulProject.config.classes import *
import IPython.display as ipd 
import matplotlib.pyplot as plt
import seaborn as sns
from SoccerFoulProject import LOGGER
from torchvision import transforms

class CFG:
    def __init__(self,
                 num_epochs=1,
                 device=torch.device('cpu'),
                 batch_size=3,
                 lr=0.001,
                 save_folder='Experiment' ,
                 gradient_accumulation_steps=1,
                 lr_scheduler=None,
                 patience=None,
                 transform=transforms.Compose([transforms.Resize((240,240)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
 
        self.num_epochs = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.transform = transform
        self.save_folder=save_folder
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.lr_scheduler=lr_scheduler
        if not(patience):
            self.patience=num_epochs
        else:
            self.patience=patience

    def to_dictionnary(self):
        return {
            'Numbre of epochs': self.num_epochs,
            'Batch Size': self.batch_size,
            'Learning rate': self.lr,
            'Gradient accumulation steps':self.gradient_accumulation_steps,
            
        }
 

def show_video(path):
    return ipd.Video(path)

def plot_results(results,path,save=False,plot=False):
    fig,axs=plt.subplots(ncols=2,nrows=int(np.floor(len(results.columns)/2)),figsize=(20,10))
    for i,col in enumerate(results.columns[1:]):
        sns.lineplot(results,x='epochs',y=col,ax=axs[i//2,i%2])
    if plot:
        plt.show()
    if save:
        fig.savefig(path.__str__()+'/results.png')

def init_folder(path):
    runs_path=Path('runs')
    if not(runs_path.exists()):
        runs_path.mkdir()
    cur_path= runs_path/path
    if cur_path.exists():
        i=1
        new_path=runs_path/(path+str(i))
        while Path(new_path).exists():
            i+=1
            new_path=runs_path/(path+str(i))
        
        Path(new_path).mkdir()
        return new_path
    else:
        return cur_path

