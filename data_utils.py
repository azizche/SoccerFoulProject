import numpy as np 
from SoccerFoulProject.utils import *
from pathlib import Path
from typing import Tuple

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
    
    def get_relative_path(self,folder_path,split):
        path_abs= Path(self.generic_path+'.mp4')
        path_r= Path(f'{folder_path}/{split}')/ path_abs.parent.name / path_abs.name
        return path_r    
    
    def read_clip(self,folder_path,split,start_pts,end_pts):
        video= read_video(self.get_relative_path(folder_path,split), pts_unit='sec', output_format='TCHW',start_pts=start_pts,end_pts=end_pts)[0]        
        return video 

class Clips:
    def __init__(self,clips:list[Clip]):
        self.clips=clips
    
    @classmethod
    def from_dictionnary(cls,data,num_views):
        res=[]
        for clip_info in data:            
            
            if len(res)<num_views:
                res.append(Clip.from_dictionnary(clip_info))
        return cls(res)
    def get_main_camera(self):
        for clip in self.clips:
            if clip.camera_type=="Main camera center":
                return clip
        LOGGER.warning('No main camera was found')

    def __len__(self):
        return len(self.clips)
    
    def read_clips(self,folder_path,split, start,end):
        return [clip.read_clip(folder_path=folder_path,split=split,start_pts=start,end_pts=end) for clip in self.clips]

def read_data(json_path):
    with open(json_path, 'r') as file:
        data=json.load(file)
    return data

def labels_to_vector(folder_path:str,split:str,num_views:int ) -> Tuple[list[Clips], list[Label]]:
    annotations= read_data(Path(folder_path) / Path(split) /Path('annotations.json'))
    video_paths=[]
    labels= []
    actions_to_skip=[]
    for action,action_info in annotations['Actions'].items():
        label=Label.from_dictionary(action_info)

        #Remove actions where action class is unknown or empty
        if label.action_class not in EVENT_DICTIONARY_action_class.keys():
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
        if len(action_info['Clips'])<num_views:
            actions_to_skip.append(action)
            continue
        if label.offence=='' or label.action_class=='' or label.severity=='':
            actions_to_skip.append(action)
            continue        
 
        video_paths.append(Clips.from_dictionnary(action_info['Clips'],num_views))
        labels.append(label)

    return video_paths, labels




        

            
    
        


