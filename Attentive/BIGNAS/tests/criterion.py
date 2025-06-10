from xnas.runner.criterion import criterion_builder
import torch
from xnas.core.config import cfg



class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing

    """ label smooth """
    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) +  self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def test_cross_entropy_smooth():
    cfg.SEARCH.LOSS_FUN = 'cross_entropy_smooth'
    cfg.SEARCH.LABEL_SMOOTH = 0.1
    criterion = criterion_builder()
    loss = criterion(preds, target)
    print(loss)
    criterion = CrossEntropyLossSmooth()
    loss = criterion(preds, target)
    print(loss)


if __name__ == '__main__':
    preds = torch.rand(size=(4,10), dtype=torch.float32)
    target = torch.randint(low=0, high=9, size=(4,), dtype=torch.int64)
    print(preds.shape)
    print(target.shape)
    test_cross_entropy_smooth()
