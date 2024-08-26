import torch


class LinearModel(torch.nn.Module):
    """
    Basic linear model for logistic regression.

    NB: This model cannot be modified, it must remain a linear model with a single layer.
    This is because in the computation of the Hessian matrix i have to hardcode the forward pass
    and there it's assumed that its a linear model.
    """

    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.output = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.output(x)
