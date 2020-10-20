import torch
from torch import optim, nn, save
from torch.utils.data import DataLoader
from torch import eye as E
from os import path, mkdir
from matplotlib import pyplot as plt

def train(model, train_ds, val_ds, train_opts, exp_dir=None):
    
    print(f"Training on {len(train_ds)} and validating on {len(val_ds)} sequences.")

    # optimizer = optim.SGD(model.parameters(), lr=train_opts['lr'], weight_decay=train_opts['weight_decay'])
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

    lr_gamma = train_opts['lr_gamma']  # 0.8
    lr_step = train_opts['lr_step'] # 50
    optimizer = optim.Adam(model.parameters(), lr=train_opts['lr'], weight_decay=train_opts['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    
    if train_opts['loss'] == 'l1':
        criterion = nn.L1Loss()
        print("criterion = nn.L1Loss()")
    elif train_opts['loss'] == 'mse':
        criterion = nn.MSELoss()
        print("criterion = nn.MSELoss()")
    else:
        print("wrong loss")


    train_dl = DataLoader(train_ds, batch_size=train_opts['batch_size'], drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=train_opts['batch_size'], drop_last=True)

    tr_loss = []
    val_loss = []

    best_val = 1e3

    for i in range(train_opts['num_epochs']):
        model.train()
        tr_l = fit(model, train_dl, criterion, optimizer)
        tr_loss.append(tr_l)

        # validation
        model.eval()
        val_l = fit(model, val_dl, criterion)
        val_loss.append(val_l)

        if exp_dir:
            if not path.exists(exp_dir):
                try:
                    mkdir(exp_dir)
                    save(model.constrains.state_dict(), path.join(exp_dir, f"model_{i + 1}.pt"))
                except FileNotFoundError:
                    pass
            else:
                save(model.constrains.state_dict(), path.join(exp_dir, f"model_{i + 1}.pt"))
            
            if (best_val > val_l):
                best_val = val_l
                print("saved best model at iter = " + str(i+1))
                save(model.constrains.state_dict(), path.join(exp_dir, f"best_model.pt"))

        print(i+1, train_opts['num_epochs'], tr_l, val_l)

        scheduler.step()
    
    plot(tr_loss, val_loss)


def fit(model, data_dl, criterion, optimizer=None):
    e_loss = 0
    # print(len(data_dl))
    for x, y in data_dl:
        pred = model(x)
        loss = criterion(pred, y)
        e_loss += loss.item()
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            clip = 1
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
    e_loss = e_loss/len(data_dl)
    return e_loss


def plot(tr_loss, val_loss):

    n = [i + 1 for i in range(len(tr_loss))]

    plt.plot(n, tr_loss, 'bs-', markersize=3, label="train")
    plt.plot(n, val_loss, 'rs-', markersize=3, label="val")
    plt.legend(loc="upper right")

    plt.show()
