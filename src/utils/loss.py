import torch
import torch.nn as nn
import torch.nn.functional as F


class VALoss(nn.Module):
    def __init__(
        self,
        loss_type,
        digitize_num,
        val_range,
        aro_range,
        lambda_ccc,
        lambda_v,
        lambda_a,
    ):
        super(VALoss, self).__init__()
        self.digitize_num = digitize_num
        self.val_ccc_loss = CCCLoss(digitize_num, range=val_range)
        self.aro_ccc_loss = CCCLoss(digitize_num, range=aro_range)
        if loss_type == "CCC_CE":
            self.val_ce_loss = Custom_CrossEntropyLoss(digitize_num, range=val_range)
            self.aro_ce_loss = Custom_CrossEntropyLoss(digitize_num, range=aro_range)
        self.lambda_ccc = lambda_ccc
        self.lambda_val = lambda_v
        self.lambda_aro = lambda_a

    def forward(self, input, target):
        loss_v = self.lambda_ccc * self.val_ccc_loss(input[:, : self.digitize_num], target[:,0])
        loss_a = self.lambda_ccc * self.aro_ccc_loss(input[:, self.digitize_num :], target[:,1])
        if hasattr(self, "aro_ce_loss") and hasattr(self, "val_ce_loss"):
            loss_v += self.val_ce_loss(input[:, : self.digitize_num], target[:,0])
            loss_a += self.aro_ce_loss(input[:, self.digitize_num :], target[:,1])

        return self.lambda_val * loss_v + self.lambda_aro * loss_a


class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1]):
        super(CCCLoss, self).__init__()
        self.digitize_num = digitize_num
        self.range = range
        if self.digitize_num != 1:
            bins = torch.linspace(*self.range, self.digitize_num)
            self.bins = bins.float().view((1, -1))

    def forward(self, input, target):
        # the target is continuous value (BS, )
        # the input is either continuous value (BS, ) or probability output(digitized)
        target = target.view(-1)
        if self.digitize_num != 1:
            input = F.softmax(input, dim=-1)
            input = (self.bins.to(input.device) * input).sum(-1)  # expectation
        input = input.view(-1)
        vx = input - torch.mean(input)
        vy = target - torch.mean(target)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2)))
            * torch.sqrt(torch.sum(torch.pow(vy, 2)))
        )
        x_m = torch.mean(input)
        y_m = torch.mean(target)
        x_s = torch.std(input)
        y_s = torch.std(target)
        ccc = (
            2
            * rho
            * x_s
            * y_s
            / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        )
        return 1 - ccc


class Custom_CrossEntropyLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1]):
        super(Custom_CrossEntropyLoss, self).__init__()
        self.digitize_num = digitize_num
        self.range = range
        assert self.digitize_num != 1
        self.edges = torch.linspace(*self.range, self.digitize_num + 1)

    def forward(self, input, target):
        # the target is continuous value (BS, )
        # the input is a probability output (digitized)
        target = target.view(-1)
        y_dig = torch.bucketize(target, self.edges.to(target.device))
        y_dig[y_dig == self.digitize_num] = self.digitize_num - 1
        target = y_dig.long()
        return F.cross_entropy(input, target)