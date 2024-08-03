import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from SoccerFoulProject.config.classes import EVENT_DICTIONARY_action_class, EVENT_DICTIONARY_offence_severity_class
from SoccerFoulProject.utils import CFG
from SoccerFoulProject.dataset import MVFoulDataset
from SoccerFoulProject.metrics import ConfusionMatrix
import pandas as pd
from SoccerFoulProject import LOGGER
from tqdm import tqdm
class MVFoulValidator:
    def __init__(self,model,dataset:MVFoulDataset,cfg:CFG):
        self.args=cfg
        self.val_dataset=dataset
        self.val_dataloader= DataLoader(self.val_dataset, self.args.batch_size)
        self.model=model
        self.loss_fn_action=torch.nn.CrossEntropyLoss()
        self.loss_fn_off_sev=torch.nn.CrossEntropyLoss()
        self.action_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_action_class.keys())
        self.off_sev_conf_matrix=ConfusionMatrix(EVENT_DICTIONARY_offence_severity_class.keys())
        self.results=pd.DataFrame(columns=['Val Action loss','Val Offence severity loss','Val Action Accuracy','Val Offence severity Accuracy'])
        

    def validation_step(self,):
        self.model.eval()

        loss_ac_all,loss_off_sev_all=0,0
        with torch.inference_mode():
            for video_clips,label in self.val_dataloader:
                video_clips=video_clips.to(self.args.device)
                pred_action,pred_offence_severity=self.model(video_clips)
                loss_ac=self.loss_fn_action(pred_action,label['Action label'].to(self.args.device))/self.args.gradient_accumulation_steps
                loss_off_sev=self.loss_fn_off_sev(pred_offence_severity,label['Offence severity label'].to(self.args.device).float())/self.args.gradient_accumulation_steps
                self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_action),dim=1),torch.argmax(label['Action label'],dim=1))
                self.action_conf_matrix.process(torch.argmax(torch.sigmoid(pred_offence_severity),dim=1),torch.argmax(label['Offence severity label'],dim=1))
                loss_ac_all+=loss_ac.detach().numpy()
                loss_off_sev_all+=loss_off_sev.detach().numpy()
            

        loss_ac_all/=len(self.val_dataloader)
        loss_off_sev_all/=len(self.val_dataloader)
        self.new_series=pd.Series({'Val Action loss':loss_ac_all,'Val Offence severity loss':loss_off_sev_all,'Val Action Accuracy':self.action_conf_matrix.compute_accuracy(),'Val Offence severity Accuracy':self.off_sev_conf_matrix.compute_accuracy()})
        self.results=pd.concat([self.results,self.new_series.to_frame().T],ignore_index=True)
        LOGGER.info(self.results.iloc[-1].to_frame().T)


    

