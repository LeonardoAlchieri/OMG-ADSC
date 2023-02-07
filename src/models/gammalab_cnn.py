from torch import concat
from torch.nn import (Conv1d, Linear, MaxPool1d, Module,
                      ReLU)


class MultiCNN(Module):
    def __init__(self, num_frames: int = 64, input_size: int = 512) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.conv1 = Conv1d(in_channels=input_size, 
                            out_channels=128, 
                            kernel_size=2, 
                            padding=0)
        self.conv2 = Conv1d(in_channels=input_size, 
                            out_channels=128, 
                            kernel_size=3, 
                            padding=0)
        self.conv3 = Conv1d(in_channels=input_size, 
                            out_channels=128, 
                            kernel_size=4, 
                            padding=0)
        self.conv4 = Conv1d(in_channels=input_size, 
                            out_channels=128, 
                            kernel_size=5, 
                            padding=0)
        
        self.maxpool1 = MaxPool1d(kernel_size=(num_frames-2+1))
        self.maxpool2 = MaxPool1d(kernel_size=(num_frames-3+1))
        self.maxpool3 = MaxPool1d(kernel_size=(num_frames-4+1))
        self.maxpool4 = MaxPool1d(kernel_size=(num_frames-5+1))
        
        self.linear1 = Linear(in_features=512, out_features=256)
        # NOTE: not use about use of relu activation, since it was not specified in the paper
        self.relu = ReLU()
        

    def forward(self, x):
        x = x.reshape(-1, x.shape[1], self.num_frames)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x1 = self.maxpool1(x1).squeeze()
        x2 = self.maxpool2(x2).squeeze()
        x3 = self.maxpool3(x3).squeeze()
        x4 = self.maxpool4(x4).squeeze()
        
        # x = x1 + x2 + x3 + x4
        x = concat([x1, x2, x3, x4], dim=1)
        # x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        
        return x
