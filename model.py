import torch


class LinearModel(torch.nn.Module):
    """
    Basic linear model for logistic regression.
    """

    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
