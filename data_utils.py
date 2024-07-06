import numpy as np 
from SoccerFoulProject.utils import *
from pathlib import Path
from typing import Tuple
def labels_to_vector(folder_path:str,split:str, ) -> Tuple[list[Clips], list[Label]]:
    annotations= read_data(Path(folder_path) / Path(split) /Path('annotations.json'))
    video_paths=[]
    labels= []
    actions_to_skip=[]
    Clip.folder_path=folder_path
    Clip.split=split
    for action,action_info in annotations['Actions'].items():
        label=Label.from_dictionary(action_info)

        #Remove actions where action class is unknown or empty
        if label.action_class=='' or label.action_class=="Don't know":
            actions_to_skip.append(action)
            continue
        
        #Remove actions that are not a dive and offence is empty
        if (label.offence=='' or label.offence=='Between') and label.action_class!="Dive":
            actions_to_skip.append(action)
            continue

        #If there is no offence, severity should be no card
        if label.offence=='No offence' and label.severity=='':
            label.severity='1'

        if (label.severity == '' or label.severity == '2.0' or label.severity == '4.0') and label.action_class != 'Dive' and label.offence != 'No offence' and label.offence != 'No Offence':
            actions_to_skip.append(action)
            continue

        if label.offence == '' or label.offence == 'Between':
            label.offence = 'Offence'

        if label.severity == '' or label.severity == '2.0' or label.severity == '4.0':
            label.severity = '1.0'
        
        if label.offence=='' or label.action_class=='' or label.severity=='':
            print('Problem')
            print('action_calss', label.action_class)
            print('severity',label.severity)
            print('offence', label.offence) 
            actions_to_skip.append(action)
            continue        

        
        video_paths.append(Clips.from_dictionnary(action_info['Clips']))
        labels.append(label)
    return video_paths, labels




        

            
    
        


