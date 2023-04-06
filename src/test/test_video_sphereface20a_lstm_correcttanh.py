import datetime
import sys
from os.path import join as join_paths
from sys import path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from numpy.random import randint
from skimage import io
from skimage.transform import resize
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm

path.append("./")
import net_sphere
from src.test import test

# FIXME: these should not be hardcoded
# Define parameters
use_cuda: bool = torch.cuda.is_available()
# use_mps: bool = torch.backends.mps.is_available()
use_mps = True

lr = 0.01
bs = 32
n_epoch = 30
lr_steps = [8, 16, 24]

gd = 20  # clip gradient
eval_freq = 3
print_freq = 18
num_worker = 8
num_seg = 16
flag_biLSTM = True

classnum = 7
correct_img_size = (112, 96, 3)


class Net(torch.nn.Module):
    def __init__(self, backbone, backbone_output_size: int = 521):
        super(Net, self).__init__()
        self.backbone = backbone
        self.linear = torch.nn.Linear(512, 2)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.avgPool = torch.nn.AvgPool2d((num_seg, 1), stride=1)
        self.LSTM = torch.nn.LSTM(
            backbone_output_size, 512, 1, batch_first=True, dropout=0.2, bidirectional=flag_biLSTM
        )  # Input dim, hidden dim, num_layer
        # self.LSTM = torch.nn.GRU(
        #     backbone_output_size, 512, 1, batch_first=True, dropout=0.2, bidirectional=flag_biLSTM
        # )  # Input dim, hidden dim, num_layer
        for name, param in self.LSTM.named_parameters():
            if "bias" in name:
                torch.nn.init.constant(param, 0.0)
            elif "weight" in name:
                torch.nn.init.orthogonal(param)

    def sequentialLSTM(self, input, hidden=None):

        input_lstm = input.view([-1, num_seg, input.shape[1]])
        batch_size = input_lstm.shape[0]
        feature_size = input_lstm.shape[2]

        self.LSTM.flatten_parameters()

        output_lstm, hidden = self.LSTM(input_lstm)
        if flag_biLSTM:
            output_lstm = (
                output_lstm.contiguous()
                .view(batch_size, output_lstm.size(1), 2, -1)
                .sum(2)
                .view(batch_size, output_lstm.size(1), -1)
            )

        # avarage the output of LSTM
        output_lstm = output_lstm.view(batch_size, 1, num_seg, -1)
        out = self.avgPool(output_lstm)
        out = out.view(batch_size, -1)
        return out

    def forward(self, x):
        x = self.backbone(x)
        x = self.sequentialLSTM(x)
        x = self.linear(x)
        
        arousal = x[:, 0]
        valence = x[:, 1]
        # valence = self.tanh(valence)
        # arousal = self.sigmoid(arousal)

        return torch.stack([arousal, valence], dim=1)


def printoneline(*argv):
    s = ""
    for arg in argv:
        s += str(arg) + " "
    s = s[:-1]
    sys.stdout.write("\r" + s)
    sys.stdout.flush()


def dt():
    return datetime.datetime.now().strftime("%H:%M:%S")


def save_model(model, filename):
    state = model.state_dict()
    torch.save(state, filename)



class OMGDataset(Dataset):
    """OMG dataset."""

    def __init__(self, txt_file, base_path, transform=None):
        self.base_path = base_path
        self.data = pd.read_csv(txt_file, sep=" ", header=0, index_col=0)
        self.data.dropna(inplace=True, how="any")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.data.iloc[idx, 0]
        utter = self.data.iloc[idx, 1]
        img_list = self.data.iloc[idx, -1]
        img_list = img_list.split(",")[:-1]
        # img_list = [int(img) for img in img_list]

        num_frames = len(img_list)
        # inspired by TSN's pytorch code
        average_duration = num_frames // num_seg
        if num_frames > num_seg:
            offsets = np.multiply(list(range(num_seg)), average_duration) + randint(
                average_duration, size=num_seg
            )
        else:
            tick = num_frames / float(num_seg)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_seg)])

        final_list = [img_list[i] for i in offsets]

        # stack images within a video in the depth dimension
        for i, ind in enumerate(final_list):
            image = io.imread(
                join_paths(self.base_path, "%s/%s/%s.png" % (vid, utter, ind))
            ).astype(np.float32)
            if correct_img_size:
                # NOTE: added here to account for possiblty different image size
                image = resize(image, correct_img_size, anti_aliasing=True)
            image = torch.from_numpy(((image - 127.5) / 128).transpose(2, 0, 1))

            if i == 0:
                images = image
            else:
                images = torch.cat((images, image), 0)

        label = torch.from_numpy(
            np.array([self.data.iloc[idx, 2], self.data.iloc[idx, 3]]).astype(
                np.float32
            )
        )

        if self.transform:
            image = self.transform(image)
        return (images, label, (vid, utter))


if __name__ == "__main__":

    test_list_path = "./support_tables/validation_list_lstm.txt"
    test_data_path: str = (
        "../Validation_Set/trimmed_faces"
    )
    # ground_truth_path: str = "./results/omg_TestVideos_WithLabels.csv"
    ground_truth_path: str = "./results/omg_ValidationVideos.csv"

    train_res_weights: str = "./pth_best/sphereface/sphereface20a_lstm_mseloss_23_0.2514_0.3820.pth"
    model_name: str = "sphereface20a_lstm_cccloss"
    
    device: str = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    

    backbone = getattr(net_sphere, "sphere20a")()
    backbone.feature = (
        True  # remove the last fc layer because we need to use LSTM first
    )

    model = Net(backbone, backbone_output_size=512)
    model.load_state_dict(torch.load(train_res_weights, map_location=device))

    if use_cuda:
        model.cuda()
    elif use_mps:
        model.to("mps", non_blocking=False)


    test_loader = DataLoader(
        OMGDataset(test_list_path, test_data_path),
        batch_size=bs,
        shuffle=False,
        num_workers=num_worker,
    )

    best_arou_ccc, best_vale_ccc = test(
        test_loader, model, model_name, 0, ground_truth_path=ground_truth_path, reshape_mode=2
    )
    print(best_arou_ccc, best_vale_ccc)
