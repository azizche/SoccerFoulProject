import torch 
from torch.utils.data import Dataset
from SoccerFoulProject.utils import *
from pathlib import Path
from SoccerFoulProject.data_utils import labels_to_vector
from SoccerFoulProject import LOGGER

class MVFoulDataset(Dataset):
    def __init__(self,folder_path:str,split:str,num_views:int,start:float,end:float, transform=None):
        if start>3 or end<3:
            LOGGER.warning('Video will not include the action timestamp')
        self.folder_path=folder_path
        self.start=start
        self.end=end
        self.video_paths,self.labels= labels_to_vector(folder_path,split,num_views,)
        self.transform=transform
        
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self,index):
        #Reading videos into lists
        videos=self.video_paths[index].read_clips(self.start,self.end)
        
        #Transforming frames in videos and slicing the videos
        if self.transform:
            videos=[self.transform(video.float()) for video in videos]

        #Stacking videos into tensor
        videos=torch.vstack([video.unsqueeze(0).float() for video in videos])

        #Permute videos to match encoder input dimension
        videos=videos.permute(0,2,1,3,4)
        label=self.labels[index]
        return videos, label.to_dictionnary()