import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=1, out_features=16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.r1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(in_features=16, out_features=64)
        
        

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x
        

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.r1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x
    