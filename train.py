import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from SoccerFoulProject.config.classes import EVENT_DICTIONARY_action_class, EVENT_DICTIONARY_offence_severity_class
from SoccerFoulProject.utils import CFG,init_folder
from SoccerFoulProject.dataset import MVFoulDataset
from SoccerFoulProject.metrics import ConfusionMatrix
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from SoccerFoulProject import LOGGER
class MVFoulTrainer:
    def __init__(self,model,dataset:MVFoulDataset,args:CFG,):
        self.args=args
        self.train_dataset=dataset
        self.model= model
        self.train_dataloader= DataLoader(self.train_dataset, self.args.batch_size)
        self.loss_fn_action=torch.nn.CrossEntropyLoss()
        self.loss_fn_off_sev=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        self.get_lr_scheduler()
        self.action_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_action_class.keys())
        self.off_sev_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_offence_severity_class.keys())
        self.save_folder=init_folder(args.save_folder)
        self.results=pd.DataFrame(columns=['Train Action loss','Train Offence severity loss','Train Action Accuracy','Train Offence severity Accuracy'])

    def get_lr_scheduler(self):
        if not(self.args.lr_scheduler):
            self.lr_scheduler=None
        else:
            lr_scheduler_map={'cosineAnnealingwarmrestarts':CosineAnnealingWarmRestarts, 'cosineannealinglr':CosineAnnealingLR, 'reducelronplateau':ReduceLROnPlateau}
            self.lr_scheduler=lr_scheduler_map[self.args.lr_scheduler](self.optimizer,)

    def train_step(self):
        loss_ac_all,loss_off_sev_all=0,0
        for step,(video_clips,label) in enumerate(self.train_dataloader):
            video_clips=video_clips.to(self.args.device)
            pred_action,pred_offence_severity=self.model(video_clips)
            
            loss_ac=self.loss_fn_action(pred_action,label['Action label'].to(self.args.device))/self.args.gradient_accumulation_steps
            loss_off_sev=self.loss_fn_off_sev(pred_offence_severity,label['Offence severity label'].to(self.args.device).float())/self.args.gradient_accumulation_steps
            self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_action),dim=1),torch.argmax(label['Action label'],dim=1))
            self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_offence_severity),dim=1),torch.argmax(label['Offence severity label'],dim=1))
            loss_ac.backward(retain_graph=True)
            loss_off_sev.backward()
            if (step+1)%self.args.gradient_accumulation_steps==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_ac_all+=loss_ac.detach().numpy()
            loss_off_sev_all+=loss_off_sev.detach().numpy()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
        loss_ac_all/=len(self.train_dataloader)
        loss_off_sev_all/=len(self.train_dataloader)
        self.new_series=pd.Series({'Train Action loss':loss_ac_all,'Train Offence severity loss':loss_off_sev_all,'Train Action Accuracy':self.action_conf_matrix.compute_accuracy(),'Train Offence severity Accuracy':self.off_sev_conf_matrix.compute_accuracy()})
        self.results=pd.concat([self.results,self.new_series.to_frame().T],ignore_index=True)
        LOGGER.info(self.results.iloc[-1].to_frame().T)
            

