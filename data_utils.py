import numpy as np 
from utils import *
from pathlib import Path
def labels_to_vector(folder_path,split):
    annotations= read_data(Path(folder_path) / Path(split) /Path('annotations.json'))
    video_paths=[]
    labels= []
    actions_to_skip=[]
    for action,action_info in annotations['Actions'].items():
        offence=action_info['Offence']
        action_class=action_info['Action class']
        severity= action_info['Severity']

        #Remove actions where action class is unknown or empty
        if action_class=='' or action_class=="Don't know":
            actions_to_skip.append(action)
            continue
        
        #Remove actions that are not a dive and offence is empty
        if offence=='' and action_class!="Dive":
            actions_to_skip.append(action)
            continue

        #If there is no offence, severity should be no card
        if offence=='No offence' and severity=='':
            severity='1'
        
        if offence=='' or action_class=='' or severity=='':
            print('Problem')

        action_clips=[]
        for clip in action_info['Clips']:
            action_clips.append(get_relative_path(clip['Url']))
        video_paths.append(action_clips)
        labels.append(Label(offence,action_class,severity))
    return video_paths, labels

        

def get_relative_path(absolute_path,folder_path,split):
    path_abs= Path(absolute_path+'.mp4')
    path_r= Path(f'{folder_path}/{split}')/ path_abs.parent.name / path_abs.name
    return path_r


        

            
    
        


