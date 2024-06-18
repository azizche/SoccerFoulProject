import torch
def train(model,data_loader,optimizer,loss,num_epochs,device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for X,y in data_loader:
            X,y=X.to(device),y.to(device)
            pred_action,pred_offence_severity=model(X)
            