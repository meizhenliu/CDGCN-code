import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, scores):
        # compute image-sentence score matrix
        # scores = get_sim(im, s)
        # diagonal = scores.diag().view(im.size(0), 1)

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5#这行代码创建了一个掩码张量，用于屏蔽具有低相似度得分的区域。创建单位矩阵，执行逐元素的比较操作，将单位矩阵中大于 0.5 的元素替换为 True，小于或等于 0.5 的元素替换为 False。这将生成一个大小相同的布尔张量
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)#I 是一个掩码张量，它应该和 cost_s 有相同的形状。I 的元素为布尔值，其中为 True 的位置表示需要被屏蔽的位置。
        cost_im = cost_im.masked_fill_(I, 0)#masked_fill_(I, 0) 是一个原地操作，它将 cost_s 中 mask 中对应位置为 True 的元素都替换为 0。这意味着在 cost_s 中，所有在 I 中对应位置为 True 的元素都被设置为 0。

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

