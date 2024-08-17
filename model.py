import torch


class LinearModel(torch.nn.Module):
    """
    Basic linear model for logistic regression.
    """

    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        # self.layer1 = torch.nn.Linear(input_dim, 4096, bias=True)
        self.output = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x = self.layer1(x)
        # x = torch.relu(x)
        return self.output(x)
