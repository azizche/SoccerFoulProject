import torch 
from torch.utils.data import Dataset
from utils import *
from torchvision.io import read_video
from pathlib import Path
from data_utils import labels_to_vector
class MVFoulDataset(Dataset):
    def __init__(self,folder_path:str,split:str):
        Label.fold
        self.folder_path=folder_path
        self.video_paths,self.labels= labels_to_vector(folder_path,split)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self,index):
        item_video_paths=[read_video(path.get_relative_path())  for paths in self.video_paths[index] for path in paths]
        item_label=self.labels[index]
        


        return item_video_paths, item_label