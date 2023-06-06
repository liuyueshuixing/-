import torch.nn as nn
import torch
class nnModel(nn.Module):
    def __init__(self,input_dim,output_dim,dropout,num_feature=1024):
        super(nnModel,self).__init__()
        self.dense1 = nn.Linear(input_dim,num_feature)
        self.dense2 = nn.Linear(num_feature,num_feature//2)
        self.dense3 = nn.Linear(num_feature//2,num_feature//4)
        self.dense4 = nn.Linear(num_feature//4,output_dim)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_feature)
        self.bn2 = nn.BatchNorm1d(num_feature//2)
        self.bn3 = nn.BatchNorm1d(num_feature//4)
    def forward(self,input):

        x = self.dense1(input)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense4(x)
        x = self.sigmoid(x)
        return x


