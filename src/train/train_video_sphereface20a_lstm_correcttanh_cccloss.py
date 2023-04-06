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
from calculateEvaluationCCC import calculateCCC
from src.utils.loss import VALoss

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
num_worker = 2
num_seg = 16
flag_biLSTM = True

classnum = 7
correct_img_size = (112, 96, 3)

model_name = 'sphereface20a_lstm_correcttanh'
loss_type = 'cccloss'


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
        )
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


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0

    for i, (inputs, targets, _) in tqdm(
        enumerate(train_loader), desc="Training batch", total=len(train_loader)
    ):
        inputs: Tensor
        targets: Tensor

        optimizer.zero_grad()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        elif use_mps:
            inputs, targets = inputs.to("mps", non_blocking=False), targets.to("mps", non_blocking=False)

        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        inputs = inputs.view((-1, 3) + inputs.size()[-2:])
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # tsn uses clipping gradient
        if gd is not None:
            total_norm = clip_grad_norm(model.parameters(), gd)
            if total_norm > gd:
                print(
                    "clippling gradient: {} with coef {}".format(
                        total_norm, gd / total_norm
                    )
                )

        train_loss += loss.data.item()

        if i % print_freq == 0:
            printoneline(
                dt(), "Epoch=%d Loss=%.4f \n" % (epoch, train_loss / (batch_idx + 1))
            )
        batch_idx += 1


def validate(val_loader, model, criterion, epoch):
    model.eval()

    err_arou = 0.0
    err_vale = 0.0

    txt_result = open("results/val_%s_%s_%d.csv" % (model_name, loss_type, epoch), "w")
    txt_result.write("video,utterance,arousal,valence\n")
    for (inputs, targets, (vid, utter)) in tqdm(val_loader, "Validation batch"):
        inputs: Tensor
        targets: Tensor
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        elif use_mps:
            inputs, targets = inputs.to("mps", non_blocking=False), targets.to("mps", non_blocking=False)

        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        inputs = inputs.view((-1, 3) + inputs.size()[-2:])
        outputs = model(inputs)

        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        err_arou += np.sum((outputs[:, 0] - targets[:, 0]) ** 2)
        err_vale += np.sum((outputs[:, 1] - targets[:, 1]) ** 2)

        for i in range(len(vid)):
            out = outputs
            txt_result.write(
                "%s,%s.mp4,%f,%f\n" % (vid[i], utter[i], out[i][0], out[i][1])
            )

    txt_result.close()

    arouCCC, valeCCC = calculateCCC(
        "./results/omg_ValidationVideos.csv",
        "results/val_%s_%s_%d.csv" % (model_name, loss_type, epoch),
    )
    return (arouCCC, valeCCC)


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

    train_list_path = "./support_tables/train_list_lstm.txt"
    val_list_path = "./support_tables/validation_list_lstm.txt"
    augmentation: bool = True
    
    train_data_path: str = (
        "../Train_Set/trimmed_faces"
    )
    device: str = 'cuda' if use_cuda else ('mps' if use_mps else 'cpu')
    validation_data_path: str = "../Validation_Set/trimmed_faces"
    
    model_path = "./model/sphere20a_20171020.pth"
    backbone = getattr(net_sphere, "sphere20a")()
    if augmentation:
        backbone.load_state_dict(torch.load(model_path))
    backbone.feature = (
        True  # remove the last fc layer because we need to use LSTM first
    )

    model = Net(backbone, backbone_output_size=512)

    if use_cuda:
        model.cuda()
    elif use_mps:
        model.to("mps", non_blocking=False)

    criterion = VALoss(loss_type='CCC', 
                       digitize_num=1, 
                       val_range=[-1,1], 
                       aro_range=[0,1], 
                       lambda_ccc=2,
                       lambda_v=1,
                       lambda_a=1)

    train_loader = DataLoader(
        OMGDataset(train_list_path, train_data_path),
        batch_size=bs,
        shuffle=True,
        num_workers=num_worker,
    )
    val_loader = DataLoader(
        OMGDataset(val_list_path, validation_data_path),
        batch_size=bs,
        shuffle=False,
        num_workers=num_worker,
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_arou_ccc, best_vale_ccc = validate(val_loader, model, criterion, 0)

    for epoch in tqdm(range(n_epoch), desc="Epoch"):
        if epoch in lr_steps:
            lr *= 0.1
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % eval_freq == 0 or epoch == n_epoch - 1:
            arou_ccc, vale_ccc = validate(val_loader, model, criterion, epoch)

            if (arou_ccc + vale_ccc) > (best_arou_ccc + best_vale_ccc):
                best_arou_ccc = arou_ccc
                best_vale_ccc = vale_ccc
                save_model(
                    model,
                    (
                        "./pth/model_%s_%s_%s_%.4f_%.4f.pth"
                        % (model_name, loss_type, epoch, arou_ccc, vale_ccc)
                    ),
                )
