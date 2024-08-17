import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
from random_fourier import RandomFourier

from model import LinearModel
from constants import DEVICE, CONFIG


def BCELossWithLogits_simple(y_pred, y_true, params=None):
    """
    Binary Cross Entropy Loss
    """
    return nn.BCEWithLogitsLoss()(y_pred, y_true)


def BCELossWithLogits_L2(y_pred, y_true, params, gamma=CONFIG["GAMMA"]):
    """
    Binary Cross Entropy Loss with L2 Regularization
    """
    params = params.squeeze()
    l2norm = torch.dot(params, params)

    return nn.BCEWithLogitsLoss()(y_pred, y_true) + gamma / 2 * l2norm


def LRLoss_simple(y_pred, y_true, params=None):
    """
    Without L2 Regularization
    """
    return torch.log(1 + torch.exp(-y_pred * y_true)).mean()


def LRLoss(y_pred, y_true, params, gamma=CONFIG["GAMMA"]):
    """
    This is the loss function taken from the paper for binary classification.
    It expects that the label should be -1 or 1, and y_pred can be in range (-inf, inf). (i.e. no sigmoid in output layer)

    If we assume that labels are 0 and 1, and use the sigmoid in output layer, then we can use the BCELoss.
    (they are equivalent, but maybe BCE loss is more stable?)
    """
    params = params.squeeze()
    l2norm = torch.dot(params, params)

    return torch.log(1 + torch.exp(-y_pred * y_true)).mean() + gamma / 2 * l2norm


def train_epoch_old(
    config,
    train_dataloader,
    net,
    optimizer,
    criterion,
    max_iters=None,
):
    """
    Train the model for one epoch
    """

    net.train()
    train_loss = 0.0
    epoch_accuracy = 0.0

    for i, (X, y) in enumerate(train_dataloader):
        if i % 100 == 0:
            print(f"Batch {i}")
        X = X.float().to(DEVICE)
        y = y.float().to(DEVICE)

        optimizer.zero_grad()

        y_pred = net(X).squeeze()
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        bools = torch.maximum(torch.zeros(len(y_pred)).to(DEVICE),
                              torch.sign(y_pred)) == y
        preds = torch.tensor([1 if b else 0 for b in bools]).float()

        epoch_accuracy += preds.mean().item()

        if max_iters is not None and i >= max_iters:
            break

    return train_loss / len(train_dataloader), epoch_accuracy / len(train_dataloader)


def train_epoch_SGD(config, train_dataloader, net, criterion, max_iters=None, RF=None):

    def backtracking_line_search(
        X, y, criterion, parameters, gradient, direction, c=0.1
    ):
        candidates = [10000, 1000, 100, 10, 1.0, 0.1, 0.01]
        for candidate in candidates:
            # Armijo - Goldstein condition
            y_hat_old = torch.matmul(
                X, torch.transpose(parameters, 0, 1)).squeeze()
            y_hat_new = torch.matmul(
                X, torch.transpose(parameters + candidate * direction, 0, 1)
            ).squeeze()

            if criterion(y_hat_new, y, parameters + candidate * direction) <= criterion(
                y_hat_old, y, parameters
            ) + candidate * c * torch.dot(direction.squeeze(), gradient.squeeze()):
                break
        # print(f"Step size: {candidate}")
        return candidate

    """Variation of train_epoch function, where I implement gradient descent by hand"""
    net.train()
    train_loss = 0.0
    epoch_accuracy = 0.0

    counter = 0

    all_y = []
    all_y_pred = []

    print("Start train dataloader for loop")
    for X, y in train_dataloader:
        if counter % 100 == 0:
            print(f"Batch {counter}")
        X = X.float()
        y = y.float().to(DEVICE)

        if RF is not None:
            RF.fit(X)
            X = RF.transform(X)

        X = torch.tensor(X).to(DEVICE).float()

        # print("Forward pass...")
        y_pred = net(X).squeeze()

        all_y.append(y.squeeze())
        all_y_pred.append(y_pred)

        # Get network parameter tensor
        parameters = torch.cat([param.view(-1) for param in net.parameters()])

        # print("Computing loss...")
        loss = criterion(y_pred, y, parameters)

        # Compute the gradient
        net.zero_grad()

        # print("Backward pass...")
        loss.backward()

        # Update the weights
        # print("Update weights...")
        with torch.no_grad():
            for param in net.parameters():
                step_size = backtracking_line_search(
                    X=X,
                    y=y,
                    criterion=criterion,
                    parameters=param,
                    gradient=param.grad,
                    direction=-param.grad,
                )
                param -= step_size * param.grad

        train_loss += loss.item()
        bools = torch.maximum(torch.zeros(len(y_pred)).to(
            DEVICE), torch.sign(y_pred.squeeze())) == y
        preds = torch.tensor([1 if b else 0 for b in bools]).float()

        epoch_accuracy += preds.mean().item()
        counter += 1

        if max_iters is not None and counter >= max_iters:
            break

    # Plot histogram of y and y_pred showing the distribution of the predictions and the true labels
    # all_y = torch.cat(all_y).cpu().numpy()
    # all_y_pred = torch.cat(all_y_pred).detach().cpu()
    # all_y_pred = torch.sign(all_y_pred.squeeze()).numpy()

    # plt.hist(all_y, bins=50, alpha=0.5, label="y")
    # plt.hist(all_y_pred, bins=50, alpha=0.5, label="y_pred")
    # plt.legend()
    # plt.show()
    return train_loss / len(train_dataloader), epoch_accuracy / len(train_dataloader)


def test_step(test_dataloader, net, criterion, RF=None):
    """
    Test the model on validation or test set
    """
    net.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    with torch.no_grad():

        for i, (X, y) in enumerate(test_dataloader):
            X = X.float()
            y = y.float().to(DEVICE)

            if RF is not None:
                RF.fit(X)
                X = RF.transform(X)

            X = torch.tensor(X).to(DEVICE).float()

            y_pred = net(X).squeeze()

            parameters = torch.cat([param.view(-1)
                                   for param in net.parameters()])
            loss = criterion(y_pred, y, parameters)

            test_loss += loss.item()

            bools = torch.maximum(torch.zeros(len(y_pred)).to(DEVICE),
                                  torch.sign(y_pred.squeeze())) == y
            preds = torch.tensor([1 if b else 0 for b in bools]).float()
            test_accuracy += preds.mean().item()

    return test_loss / len(test_dataloader), test_accuracy / len(test_dataloader)


def train(
    config,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    net,


):
    """
    Centralized training loop
    """

    # TODO: Check if we can actually use SGD or we need to do things by hand
    # (e.g. in the paper they use the line search method to determine the step size)
    # optimizer = torch.optim.SGD(
    #     net.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["GAMMA"]
    # )
    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.01, weight_decay=config["GAMMA"]
    )

    # criterion = LRLoss_simple
    criterion = BCELossWithLogits_L2

    # Define Random Fourier transformation object if we are using it
    if config["USE_RANDOM_FOURIER"]:
        RF = RandomFourier(n_components=config["N_FOURIER_FEATURES"])
    else:
        RF = None

    best_valid_loss = float("inf")
    run_results_dir = os.path.join(config["RESULTS_DIR"], config["MODEL_NAME"])
    os.makedirs(run_results_dir, exist_ok=True)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(1, config["EPOCHS"] + 1):
        print(f"Start epoch {epoch}")
        # train_loss, train_accuracy = train_epoch(
        #     config, train_dataloader, net, optimizer, criterion
        # )

        train_loss, train_accuracy = train_epoch_SGD(
            config=config, train_dataloader=train_dataloader, criterion=criterion, net=net, RF=RF)
        # train_loss, train_accuracy = train_epoch_old(
        #     config=config,
        #     train_dataloader=train_dataloader,
        #     net=net,
        #     optimizer=optimizer,
        #     criterion=criterion
        # )

        valid_loss, valid_accuracy = test_step(
            valid_dataloader, net, criterion, RF)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(
            {
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": valid_loss,
                "Validation Accuracy": valid_accuracy,
            }
        )

        if valid_loss < best_valid_loss and config["WANDB_MODE"] == "online":
            best_valid_loss = valid_loss
            # Save the model
            torch.save(
                net.state_dict(), os.path.join(run_results_dir, "model_weights.pt")
            )

            # Save online
            # wandb.save(os.path.join(run_results_dir, "model_weights.pt"))

    # Test phase
    test_loss, test_accuracy = test_step(test_dataloader, net, criterion)
    # wandb.log({"Test Loss": test_loss, "Test RMSE": test_rmse})

    # run.finish()
    return (
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        test_loss,
        test_accuracy,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from constants import CONFIG, DEVICE
    from dataset import EpsilonDataset
    from model import LinearModel
    from utils import SPLIT

    EpsilonDataset.generate_agent_splits(CONFIG, seed=0)

    client_idx = 0
    train_dataset = EpsilonDataset(
        config=CONFIG, client_idx=client_idx, split=SPLIT.TRAIN
    )
    valid_dataset = EpsilonDataset(
        config=CONFIG, client_idx=client_idx, split=SPLIT.VALIDATION
    )
    test_dataset = EpsilonDataset(
        config=CONFIG, client_idx=client_idx, split=SPLIT.TEST
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        drop_last=True,
        num_workers=os.cpu_count(),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        drop_last=True,
        num_workers=os.cpu_count(),
    )

    input_dim = CONFIG["N_FOURIER_FEATURES"] if CONFIG["USE_RANDOM_FOURIER"] else 2000
    net = LinearModel(input_dim=input_dim).float().to(DEVICE)

    (
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        test_loss,
        test_accuracy,
    ) = train(CONFIG, train_dataloader, valid_dataloader, test_dataloader, net)

    # Make two subplots, one with the train and validation losses, another with train and validation accuracies
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(valid_losses, label="Validation Loss")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(train_accuracies, label="Train Accuracy")
    axs[1].plot(valid_accuracies, label="Validation Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    # save the plot
    plt.savefig(
        os.path.join(CONFIG["RESULTS_DIR"],
                     CONFIG["MODEL_NAME"], "test_plot.png")
    )

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
