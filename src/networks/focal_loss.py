import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     self.alpha = Variable(torch.ones(class_num, 1))
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        '''
        input: (B)
        target: (B)
        '''

        P = F.sigmoid(inputs)
        pt = (1-P) * targets + P * (1-targets)
        focal_weight = (self.alpha * targets + (1-self.alpha) * (1-targets)) * torch.pow(pt, self.gamma)

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * focal_weight

        # loss = loss / N
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss