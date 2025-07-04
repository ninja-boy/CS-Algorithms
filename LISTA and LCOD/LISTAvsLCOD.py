import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.set_default_dtype(torch.float64)

class SimulatedData(Data.Dataset):
    def __init__(self, x, H, s):
        self.x = x
        self.s = s
        self.H = H

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :]
        H = self.H
        s = self.s[idx, :]
        return x, H, s

def create_data_set(H, n, m, k, N=1000, batch_size=512, signal_dev=0.5, noise_dev=0.01):

    # Initialization
    x = torch.zeros(N, n)
    s = torch.zeros(N, m)

    # Create signals
    for i in range(N):
        # Create a k-sparsed signal s
        index_k = np.random.choice(m, k, replace=False)
        peaks = signal_dev * np.random.randn(k)

        s[i, index_k] = torch.from_numpy(peaks).to(s)

        # X = Hs+w
        x[i, :] = H @ s[i, :] + noise_dev * torch.randn(n)

    simulated = SimulatedData(x=x, H=H, s=s)
    data_loader = Data.DataLoader(dataset=simulated, batch_size=batch_size, shuffle=True)
    return data_loader

N = 1000 # number of samples
n = 150 # dim(x)
m = 200 # dim(s)
k = 4 # k-sparse signal

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)

# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)

x_exm, _, s_exm =test_loader.dataset.__getitem__(5)
plt.figure(figsize=(8, 8)) 
plt.subplot(2, 1, 1) 
plt.plot(x_exm, label = 'observation' ) 
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend( )
plt.subplot (2, 1, 2) 
plt.plot(s_exm, label = 'sparse signal', color='k')  
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend( )
plt.show()

def train(model, train_loader, valid_loader, num_epochs=50):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """
    # Initialization
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=5e-05,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )
    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,))
    # Main loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for step, (b_x, b_H, b_s) in enumerate(train_loader):
            s_hat, _ = model.forward(b_x)
            loss = F.mse_loss(s_hat, b_s, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.data.item()
        loss_train[epoch] = train_loss / len(train_loader.dataset)
        scheduler.step()

        # validation
        model.eval()
        test_loss = 0
        for step, (b_x, b_H, b_s) in enumerate(valid_loader):
            # b_x, b_H, b_x = b_x.cuda(), b_H.cuda(), b_s.cuda()
            s_hat, _ = model.forward(b_x)
            test_loss += F.mse_loss(s_hat, b_s, reduction="sum").data.item()
        loss_test[epoch] = test_loss / len(valid_loader.dataset)
        # Print
        if epoch % 10 == 0:
            print(
                "Epoch %d, Train loss %.8f, Validation loss %.8f"
                % (epoch, loss_train[epoch], loss_test[epoch])
            )

    return loss_test

class LISTA_Model(nn.Module):
    def __init__(self, n, m, T=6, rho=1.0, H=None):
        super(LISTA_Model, self).__init__()
        self.n, self.m = n, m
        self.H = H
        self.T = T  # ISTA Iterations
        self.rho = rho 
        self.A = nn.Linear(n, m, bias=False)  # Weight Matrix
        self.B = nn.Linear(m, m, bias=False)  # Weight Matrix
        # ISTA Stepsizes eta
        self.beta = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True) #
        self.mu = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)
        # Initialization
        if H is not None:
            self.A.weight.data = H.t()
            self.B.weight.data = H.t() @ H

        # A and B should not be trained
        for param in self.A.parameters():
            param.requires_grad = False
        for param in self.B.parameters():
            param.requires_grad = False
        

    def _shrink(self, s, beta):
        return beta * F.softshrink(s / beta, lambd=self.rho)

    def forward(self, x, s_gt=None):
        mse_vs_itr = []

        s_hat = self._shrink(self.mu[0, :, :] * self.A(x), self.beta[0, :, :])
        for i in range(1, self.T + 1):
            s_hat = self._shrink(s_hat - self.mu[i, :, :] * self.B(s_hat) + self.mu[i, :, :] * self.A(x),
                                 self.beta[i, :, :], )
            
            # Aggregate each iteration's MSE loss
            if s_gt is not None:
                mse_vs_itr.append(F.mse_loss(s_hat.detach(), s_gt.detach(), reduction="sum").data.item())

        return s_hat, mse_vs_itr
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCOD_Model(nn.Module):
    def __init__(self, n, m, T=6, rho=1.0, H=None):
        super(LCOD_Model, self).__init__()
        self.T = T
        self.rho = rho

        self.A = nn.Linear(n, m, bias=False)  # A = H^T

        self.beta = nn.Parameter(torch.ones(T + 1, 1, 1))
        self.mu = nn.Parameter(torch.ones(T + 1, 1, 1))

        if H is not None:
            self.A.weight.data = H.T.detach().clone()

    def _shrink(self, s, beta):
        return beta * F.softshrink(s / beta, lambd=self.rho)

    def forward(self, x, s_gt=None):
        mse_vs_itr = []
        s_hat = torch.zeros(x.shape[0], self.A.out_features, dtype=x.dtype, device=x.device)

        for i in range(self.T + 1):
            beta_i = F.softplus(self.beta[i])
            mu_i = F.softplus(self.mu[i])

            residual = x - s_hat @ self.A.weight
            s_hat = s_hat + mu_i * self._shrink(self.A(residual), beta_i)

            if s_gt is not None:
                with torch.no_grad():
                    mse_vs_itr.append(F.mse_loss(s_hat, s_gt, reduction="sum").item())

        return s_hat, mse_vs_itr

def lista_apply(train_loader, test_loader, T, H):
    n = H.shape[0]
    m = H.shape[1]
    lista = LISTA_Model(n=n, m=m, T=T, H=H)
    train(lista, train_loader, test_loader)
    
    # Extract all samples and calculate MSE for each iteration
    s_gt, x = test_loader.dataset.s, test_loader.dataset.x
    _, mse_vs_iter = lista(x, s_gt=s_gt)

    return np.array(mse_vs_iter)/len(test_loader.dataset)

def lcod_apply(train_loader, test_loader, T, H):
    n = H.shape[0]
    m = H.shape[1]
    lcod = LCOD_Model(n=n, m=m, T=T, H=H)
    train(lcod, train_loader, test_loader)
    
    # Extract all samples and calculate MSE for each iteration
    s_gt, x = test_loader.dataset.s, test_loader.dataset.x
    _, mse_vs_iter = lcod(x, s_gt=s_gt)

    return np.array(mse_vs_iter)/len(test_loader.dataset)

T = 10

lista_mse_vs_iter = lista_apply(train_loader, test_loader, T, H)
lcod_mse_vs_iter = lcod_apply(train_loader, test_loader, T, H)

fig = plt.figure()
plt.plot(range(T+1), lcod_mse_vs_iter, label='LCOD', color='b', linewidth=2)
plt.plot(range(T), lista_mse_vs_iter, label='LISTA', color='r', linewidth=2)
plt.xlabel('Number of iterations', fontsize=10)
plt.ylabel('MSE', fontsize=10)
plt.yscale("log")
plt.legend()
plt.show()