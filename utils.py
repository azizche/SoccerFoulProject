import json

def read_data(json_path):
    with open(json_path, 'r') as file:
        json.load(file)

class Label:
    def __init__(self,offence,action_class,severity):
        self.offence=offence
        self.action_class=action_class
        self.severity=severity
    
