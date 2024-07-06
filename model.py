from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
import torch
class MVFoulModel(nn.Module):
    def __init__(self,video_encoder_name='r3d_18', clip_aggregation='mean',feat_dim=400):
        super(MVFoulModel,self).__init__()
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
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.offence_classification_net=nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
            nn.Linear(feat_dim, feat_dim),
        )


    

    def forward(self, clips):
        #compute video features
        all_clip_features=self.video_encoder(clips)        
        #aggregate all clips' features
        if self.clip_agregation=='mean':
            action_features= torch.mean(all_clip_features,dim=0) 
        elif self.clip_agregation=='max':
            action_features=torch.max(all_clip_features,dim=0)
        else:
            print('problem should be mean or max')
        pred_action=self.action_classifcation_net(action_features)
        pred_offence_severity=self.offence_classification_net(action_features)



        return pred_action,pred_offence_severity

