from torch import nn
import torch
from torch.nn import functional as F

class lossfun(nn.Module):
    def __init__(self, size_average=True):
        super(lossfun, self).__init__()
        self.size_average = size_average
        self.gamma = 0.25

        self.mu=1.0
        self.sigma=0.05
        self.base = 1.0
        self.right_side = True
        self.both_sides = False
    def focal_loss(self, preds_softmax, preds_logsoft):
        # Here, the alpha value is set as 1 in focal loss due to balanced class
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma): (1-pt)^γ *log(pt)
        return loss

    def gaussian_loss(self, preds_softmax, preds_logsoft):
        if self.right_side:
            # 1 + exp(-(pt-mu)^2/(2*sigma^2))
            y = self.base + torch.exp(-(preds_softmax - self.mu) ** 2 / (2 * self.sigma**2))

        if self.both_sides:
            # 1 + exp(-(pt-mu1)^2/(2*sigma1^2)) + exp(-(pt-mu2)^2/(2*sigma2^2))
            y = self.base + torch.exp(-(preds_softmax - self.mu) ** 2 / (2 * self.sigma ** 2)) + torch.exp(-(preds_softmax - 0.0) ** 2 / ((self.sigma/2)**2))

        loss = -torch.mul(y, preds_logsoft)

        return loss

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = F.softmax(preds, dim=1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        lossvalue = self.gaussian_loss(preds_softmax, preds_logsoft)

        if self.size_average:
            loss = lossvalue.mean()
        else:
            loss = lossvalue.sum()
        return loss


if __name__ == '__main__':
    pred = torch.randn((3, 5))
    print("pred:", pred)
    label = torch.tensor([2, 3, 4])
    print("label:", label)
    loss_fn = lossfun(gamma=2, num_classes=7)
    loss = loss_fn(pred, label)
    print(loss)
