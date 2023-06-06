import torch.optim as optim
from nn_Model import nnModel
import torch.nn as nn
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
def train_nn(args,train_loader,device,input_dim,output_dim,X_test,Y_test):
    Epoch = 150
    model = nnModel(input_dim,output_dim,dropout=0.3)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bceloss = nn.BCELoss()
    for e in range(Epoch):
        model.train()
        for batch_idx,batch in enumerate(train_loader):
            optimizer.zero_grad()
            # print(batch)
            # print(batch[0])
            X_embedding = batch[0].to(device)
            Y_label = batch[1].to(device)
            output = model(X_embedding)
            loss = bceloss(output,Y_label)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_prob_test = model(torch.Tensor(X_test).cuda(device))
            loss_test = bceloss(y_prob_test,torch.Tensor(Y_test).cuda(device))
        print('Epoch '+str(e)+'Train: ',loss,'\t'+'Test:\t',loss_test)
    model.eval()
    with torch.no_grad():
        y_prob = model(torch.Tensor(X_test).cuda(device))
    return y_prob
