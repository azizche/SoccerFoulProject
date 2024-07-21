import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from SoccerFoulProject.config.classes import EVENT_DICTIONARY_action_class, EVENT_DICTIONARY_offence_severity_class
from SoccerFoulProject.utils import CFG,init_folder
from SoccerFoulProject.dataset import MVFoulDataset
from SoccerFoulProject.metrics import ConfusionMatrix
import pandas as pd

class MVFoulTrainer:
    def __init__(self,model,dataset:MVFoulDataset,args:CFG,):
        self.args=args
        self.train_dataset=dataset
        self.model= model
        self.train_dataloader= DataLoader(self.train_dataset, self.args.batch_size)
        self.loss_fn_action=torch.nn.CrossEntropyLoss()
        self.loss_fn_off_sev=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        self.action_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_action_class.keys())
        self.off_sev_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_offence_severity_class.keys())
        self.save_folder=init_folder(args.save_folder)
        self.results=pd.DataFrame(columns=['Train Action loss','Train Offence severity loss','Train Action accuracy','Train Offence severity accuracy'])
        
    def train_step(self):
        loss_ac_all,loss_off_sev_all=0,0
        for video_clips,label in self.train_dataloader:
            video_clips=video_clips.to(self.args.device)
            pred_action,pred_offence_severity=self.model(video_clips)
            loss_ac=self.loss_fn_action(pred_action,label['Action label'].to(self.args.device))
            loss_off_sev=self.loss_fn_off_sev(pred_offence_severity,label['Offence severity label'].to(self.args.device).float())
            self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_action),dim=1),torch.argmax(label['Action label'],dim=1))
            self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_offence_severity),dim=1),torch.argmax(label['Offence severity label'],dim=1))
            self.optimizer.zero_grad()
            loss_ac.backward(retain_graph=True)
            loss_off_sev.backward()
            self.optimizer.step()
            loss_ac_all+=loss_ac.detach().numpy()
            loss_off_sev_all+=loss_off_sev.detach().numpy()
        

        loss_ac_all/=len(self.train_dataloader)
        loss_off_sev_all/=len(self.train_dataloader)
        self.new_series=pd.Series({'Train Action loss':loss_ac_all,'Train Offence severity loss':loss_off_sev_all,'Train Action accuracy':self.action_conf_matrix.compute_accuracy(),'Train Offence severity accuracy':self.off_sev_conf_matrix.compute_accuracy()})
        self.results=pd.concat([self.results,self.new_series],ignore_index=True)

            

