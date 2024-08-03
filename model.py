from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
import torch
from SoccerFoulProject.config.classes import *
from SoccerFoulProject.train import MVFoulTrainer
from SoccerFoulProject.validator import MVFoulValidator
from SoccerFoulProject.utils import plot_results
from torchvision.io import read_video
import pandas as pd
import json
from pathlib import Path
from SoccerFoulProject import LOGGER
from tqdm import tqdm
class Model(nn.Module):
    def __init__(self,video_encoder_name='r3d_18', clip_aggregation='mean',feat_dim=100):
        super(Model,self).__init__()
        if video_encoder_name== 'r3d_18':
            self.video_encoder= r3d_18(weights= R3D_18_Weights.DEFAULT)
        elif video_encoder_name=='mc3_18':
            self.video_encoder= mc3_18(weights=MC3_18_Weights.DEFAULT)
        elif video_encoder_name=='r2plus1d_18':
            self.video_encoder= r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        elif video_encoder_name=='s3d':
            self.video_encoder=s3d(weights= S3D_Weights.DEFAULT)
        elif video_encoder_name=='mvit_v2_s':
            self.video_encoder=mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        elif video_encoder_name=='mvit_v1_b':
            self.video_encoder=mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
        self.clip_agregation=clip_aggregation
        self.action_classifcation_net=nn.Sequential(
            nn.LayerNorm(400),
            nn.Linear(400, feat_dim),
            nn.Sigmoid(),
            nn.Linear(feat_dim, len(EVENT_DICTIONARY_action_class)),
        )
        self.offence_classification_net=nn.Sequential(
            nn.LayerNorm(400),
            nn.Linear(400, feat_dim),
            nn.Sigmoid(),
            nn.Linear(feat_dim, 4),
            
        )
        self.cfg=None
        self.results=pd.DataFrame(columns=['epochs','Train Action loss','Val Action loss','Train Offence severity loss','Val Offence severity loss','Train Action Accuracy','Val Action Accuracy','Train Offence severity Accuracy','Val Offence severity Accuracy'])

    

    def forward(self, batch_clips):
        #compute video features
        batched_pred_action = torch.empty(0, len(EVENT_DICTIONARY_action_class),)
        batched_pred_offence_severity = torch.empty(0, len(EVENT_DICTIONARY_offence_severity_class),)
        for clips in batch_clips:
            all_clip_features=self.video_encoder(clips)        
            #aggregate all clips' features
            if self.clip_agregation=='mean':
                action_features= torch.mean(all_clip_features,dim=0) 
            elif self.clip_agregation=='max':
                action_features,_=torch.max(all_clip_features,dim=0)
            else:
                raise ValueError('Clip aggreagation method should be mean or max')
                
            pred_action=self.action_classifcation_net(action_features)
            pred_offence_severity=self.offence_classification_net(action_features)
            batched_pred_action =torch.cat((batched_pred_action,pred_action.unsqueeze(0)),dim=0)
            batched_pred_offence_severity =torch.cat((batched_pred_offence_severity,pred_offence_severity.unsqueeze(0)),dim=0)
        return batched_pred_action,batched_pred_offence_severity
    
    def do_train(self, train_dataset,val_dataset,cfg):
        trainer=MVFoulTrainer(self,train_dataset,cfg)
        self.cfg=cfg
        validator=MVFoulValidator(self,val_dataset,cfg)
        best_averaged_accuracy=0
        num_epochs_with_no_improvement=0
        for epoch in tqdm(range(cfg.num_epochs)):
            trainer.train_step()
            validator.validation_step()

            #Appending training and validation results
            new_col = pd.concat([pd.Series({'epochs': epoch+1}), trainer.new_series, validator.new_series])
            self.results.loc[epoch]=new_col

            #saving model's last weights
            self.save(trainer.save_folder/Path('weights')/Path('last.pth'))

            #Implementing early stopping and saving the best model's params
            curr_overall_accuracy=(self.results.loc[epoch,'Val Action Accuracy'] + self.results.loc[epoch,'Val Offence severity Accuracy'])/2
            if best_averaged_accuracy<curr_overall_accuracy:
                best_averaged_accuracy=curr_overall_accuracy
                self.save(trainer.save_folder/Path('weights')/Path('best.pth'))
                num_epochs_with_no_improvement=0
            else:
                num_epochs_with_no_improvement+=1
            if num_epochs_with_no_improvement==cfg.patience:
                LOGGER.info('Model has stopeed learning after %d epochs with no improvement in accuracy',cfg.patience)
                break
        
        #Plotting results
        plot_results(self.results,path=trainer.save_folder,save=True,plot=False)
    
        #saving the hyperparameters
        with open(trainer.save_folder.__str__()+'/hyperparameters.json','w') as f:
            json.dump(cfg.to_dictionnary(),f)

    def save(self,path:Path):            
        if not(path.parent.exists()):
            Path(path.parent).mkdir(parents=True)
        torch.save(self.state_dict(),path.__str__())
    
    def load(self,path):
        weights=torch.load(path)
        self.load_state_dict(weights)
    

    def predict(self,path,start,end):
        video,_,_=read_video(path,pts_unit='sec', output_format='TCHW',start_pts=start,end_pts=end)
        if self.cfg and self.cfg.transform:
            video=self.cfg.transform(video.float())
        video=video.permute(1,0,2,3)
        self.eval()
        with torch.inference_mode():
            pred_action,pred_offence_severity=self(video.unsqueeze(0).unsqueeze(0))
        action=INVERSE_EVENT_DICTIONARY_action_class[ torch.argmax(torch.sigmoid(pred_action.squeeze())).item()]
        off_severity=INVERSE_EVENT_DICTIONARY_offence_severity_class[ torch.argmax(torch.sigmoid(pred_offence_severity.squeeze())).item()]
        return action,off_severity


