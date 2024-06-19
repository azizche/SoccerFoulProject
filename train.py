import torch
import torch.utils
import torch.utils.data
def step(model:torch.nn.Module,
         data_loader:torch.utils.data.DataLoader,
         optimizer:torch.optim.Optimizer,
         loss_fn_action:torch.nn.Module,
         loss_fn_off_sev:torch.nn.Module,
         num_epochs:int,
         device:torch.device,
         train:bool):
    model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    for epoch in range(num_epochs):
        loss_ac_all,loss_off_sev_all=0,0
        for video_clips,label in data_loader:
            video_clips,y=video_clips.to(device),y.to(device)
            pred_action,pred_offence_severity=model(video_clips)
            loss_ac=loss_fn_action(pred_action,label.action)
            loss_ac_all+=loss_ac
            loss_off_sev_all+=loss_off_sev
            loss_off_sev=loss_fn_off_sev(pred_offence_severity,label.get_offence_severity_label())
            if train:
                optimizer.zero_grad()
                loss_ac.backward()
                loss_off_sev.backward()
                optimizer.step()
        loss_ac_all/=len(data_loader)
        loss_off_sev_all/=len(data_loader)
            


        

