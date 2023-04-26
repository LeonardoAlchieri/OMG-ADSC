import datetime
import sys
from os.path import join as join_paths
from sys import path

import numpy as np
import pandas as pd
import torch
from numpy.random import randint
from skimage import io
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

path.append("./")
from src.models.swine import Head, swin_3d_tiny
from src.test import test

# FIXME: these should not be hardcoded
# Define parameters
use_cuda: bool = torch.cuda.is_available()
# use_mps: bool = torch.backends.mps.is_available()
use_mps = False

lr = 0.01
bs = 32
n_epoch = 30
lr_steps = [8, 16, 24]

gd = 20  # clip gradient
eval_freq = 3
print_freq = 20
num_worker = 10
num_seg = 16

classnum = 7
correct_img_size = (112, 96, 3)


class Net(torch.nn.Module):
    def __init__(self, backbone):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head_val = Head(in_channels=768, hidden_channels=64)
        self.head_aro = Head(in_channels=768, hidden_channels=64)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # BxFxCxHxW -> BxFxCxHxW
        x = self.backbone(x)
        x_val = self.head_val(x).reshape(
            -1,
        )
        x_aro = self.head_aro(x).reshape(
            -1,
        )
        return torch.stack([x_val, x_aro], dim=1)


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
    test_data_path: str = "/data/leonardo/OMGEmotionChallenge/Validation_Set/trimmed_faces"
    ground_truth_path: str = "./results/omg_ValidationVideos.csv"

    train_res_weights: str = "./pth_best/swine3dtiny/swine3dtiny_transformer_ccc_NOPRETRAIN_11_0.0370_0.1762.pth"

    model_name = "swine3dtiny_transformer_cccloss_NOPRETRAIN"

    device: str = "cuda" if use_cuda else ("mps" if use_mps else "cpu")

    backbone = swin_3d_tiny()

    model = Net(backbone)
    model.load_state_dict(torch.load(train_res_weights, map_location=device))

    if use_cuda:
        model.cuda()
    elif use_mps:
        model.to("mps")

    test_loader = DataLoader(
        OMGDataset(test_list_path, test_data_path),
        batch_size=bs,
        shuffle=False,
        num_workers=num_worker,
    )

    best_arou_ccc, best_vale_ccc = test(
        val_loader=test_loader,
        model=model,
        model_name=model_name,
        epoch=0,
        ground_truth_path=ground_truth_path,
        reshape_mode=1,
    )
    print(best_arou_ccc, best_vale_ccc)
