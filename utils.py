import json
from pathlib import Path
from torchvision.io import read_video

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
        if self.offence=='No offence':
            return 0
        if self.offence=='Offence':
            return int(self.severity[0]) + 1




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
    
    def read_clip(self,action_ts_offset):
        video= read_video(self.get_relative_path(), pts_unit='pts', output_format='TCHW',)[0]
        #transforming the video
        
        return video 

class Clips:
    def __init__(self,clips:list[Clip]):
        self.clips=clips
    
    @classmethod
    def from_dictionnary(cls,data):
        return cls([Clip.from_dictionnary(clip_info) for clip_info in data])
    
    def get_main_camera(self):
        for clip in self.clips:
            if clip.camera_type=="Main camera center":
                return clip
        #TODO: add warning if no main camera found

    def __len__(self):
        return len(self.clips)
    
    def read_clips(self, action_ts_offset):
        return [clip.read_clip(action_ts_offset) for clip in self.clips]