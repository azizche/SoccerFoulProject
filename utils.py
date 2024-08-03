import json
from pathlib import Path
from torchvision.io import read_video
import numpy as np
import torch
from SoccerFoulProject.config.classes import *
import IPython.display as ipd 
import matplotlib.pyplot as plt
import seaborn as sns
import logging 
from SoccerFoulProject import LOGGER

def read_data(json_path):
    with open(json_path, 'r') as file:
        data=json.load(file)
    return data
    
        

def get_relative_path(absolute_path,folder_path,split):
    path_abs= Path(absolute_path+'.mp4')
    path_r= Path(f'{folder_path}/{split}')/ path_abs.parent.name / path_abs.name
    return path_r

#TODO: add seperate files for testing
class Label:
    def __init__(self,offence,action_class,severity,contact,bodypart,upperBodypart, multipleFouls, tryToplay, tourchBall, handball, handballOffence):
        self.offence=offence
        self.action_class=action_class
        self.severity=severity
        self.contact=contact
        self.bodypart=bodypart
        self.upperBodypart=upperBodypart
        self.multipleFouls=multipleFouls
        self.tryToplay=tryToplay
        self.touchBall=tourchBall
        self.handball=handball
        self.handballOffence=handballOffence
        
    @classmethod
    def from_dictionary(cls,data):
        return cls(offence=data['Offence'], 
                     action_class=data['Action class'], 
                     severity=data['Severity'], 
                     contact=data['Contact'], 
                     bodypart=data['Bodypart'],
                     upperBodypart=data['Upper body part'],
                     multipleFouls=data['Multiple fouls'],
                     tryToplay=data['Try to play'],
                     handball=data['Handball'],
                     handballOffence=data['Handball offence'],
                     tourchBall=data['Touch ball'],
                     )
    def get_offence_severity_label(self):
        offence_severit_label=torch.zeros((4,))
        if self.offence=='No offence':
            index=0
        if self.offence=='Offence':
            index=int(self.severity[0])//2 +1
        offence_severit_label[index]=1
        return offence_severit_label
    def get_action_label(self):
        one_hot_encoded_label=torch.zeros((len(EVENT_DICTIONARY_action_class),))
        one_hot_encoded_label[EVENT_DICTIONARY_action_class[self.action_class]]=1
        return one_hot_encoded_label
    def to_dictionnary(self):
        return {'Offence severity label':self.get_offence_severity_label(),
                'Action label':self.get_action_label()}




class Clip:
    folder_path=''
    split=''
    def __init__(self,generic_path,camera_type,action_timestamp,replay_speed):
        self.generic_path=generic_path
        self.camera_type=camera_type
        self.action_timestamp=int(action_timestamp)
        self.replay_speed=replay_speed
    
    @classmethod
    def from_dictionnary(cls,data):
        return cls(generic_path=data['Url'],
                   camera_type=data['Camera type'],
                   action_timestamp=int(data['Timestamp']),
                   replay_speed=data['Replay speed'])
    
    def get_relative_path(self):
        path_abs= Path(self.generic_path+'.mp4')
        path_r= Path(f'{Clip.folder_path}/{Clip.split}')/ path_abs.parent.name / path_abs.name
        return path_r    
    
    def read_clip(self,start_pts,end_pts):
        video= read_video(self.get_relative_path(), pts_unit='sec', output_format='TCHW',start_pts=start_pts,end_pts=end_pts)[0]        
        return video 

class Clips:
    def __init__(self,clips:list[Clip]):
        self.clips=clips
    
    @classmethod
    def from_dictionnary(cls,data,num_views):
        res=[]
        main_camera_found=False
        for clip_info in data:
            
            if clip_info['Camera type']=="Main camera center":
                main_camera_found=True
                res.append(Clip.from_dictionnary(clip_info))
            else:
                if not(main_camera_found) and len(res)<(num_views-1):
                    res.append(Clip.from_dictionnary(clip_info))
                elif main_camera_found and len(res)<num_views:
                    res.append(Clip.from_dictionnary(clip_info))
            if len(res)==num_views:
                return cls(res)   
    
    def get_main_camera(self):
        for clip in self.clips:
            if clip.camera_type=="Main camera center":
                return clip
        LOGGER.warning('No main camera was found')

    def __len__(self):
        return len(self.clips)
    
    def read_clips(self, start,end):
        return [clip.read_clip(start_pts=start,end_pts=end) for clip in self.clips]

class CFG:
    def __init__(self,
                 num_epochs: int,
                 device=torch.device('cpu'),
                 batch_size=3,
                 lr=0.001,
                 save_folder='Experiment' ,
                 gradient_accumulation_steps=1,
                 lr_scheduler=None,
                 patience=None,
                 transform=None):
 
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

