import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


prices = pd.read_csv('prices.csv')
prices = prices.T
price_tensor = torch.tensor(prices.values, dtype=torch.float32)
X_train = price_tensor[:, :500]
X_test = price_tensor[:, 500:]

daily_diff = np.mean(np.square(prices.iloc[:, 500:].diff(axis=1).dropna(axis=1)),axis=1)
print(daily_diff)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f"device: {device}")

sequence_size = 50
sequence_train = [(i*sequence_size,(i+1)*sequence_size) for i in range(0,10)]
sequence_test = [(i*sequence_size,(i+1)*sequence_size) for i in range(0,5)]
hidden_size = 51

# define the GRU


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wz = nn.Parameter(torch.empty(hidden_size, input_size, requires_grad=True))
        nn.init.xavier_uniform_(self.Wz, gain=1.0)

        self.Uz = nn.Parameter(torch.empty(hidden_size, hidden_size, requires_grad=True))
        nn.init.orthogonal_(self.Uz, gain=1.0)

        self.Wr = nn.Parameter(torch.empty(hidden_size, input_size, requires_grad=True))
        nn.init.xavier_uniform_(self.Wr, gain=1.0)

        self.Ur = nn.Parameter(torch.empty(hidden_size, hidden_size, requires_grad=True))
        nn.init.orthogonal_(self.Ur, gain=1.0)

        self.Wh = nn.Parameter(torch.empty(hidden_size, input_size, requires_grad=True))
        nn.init.xavier_uniform_(self.Wh, gain=1.0)

        self.Uh = nn.Parameter(torch.empty(hidden_size, hidden_size, requires_grad=True))
        nn.init.orthogonal_(self.Uh, gain=1.0)

        self.Wy = nn.Parameter(torch.empty(hidden_size, hidden_size, requires_grad=True))
        nn.init.xavier_uniform_(self.Wy, gain=1.0)

        self.br = nn.Parameter(torch.zeros(hidden_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.bz = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.by = nn.Parameter(torch.tensor(np.mean(prices,axis=1).values, dtype=torch.float32))

    def forward(self, prices_t, h):
        update_gate = torch.sigmoid((self.Wz @ prices_t) + (self.Uz @ h) + self.bz)
        reset_gate = torch.sigmoid((self.Wr @ prices_t) + (self.Ur @ h) + self.br)
        candidate_act_vector = torch.tanh((self.Wh @ prices_t) + (self.Uh @ (torch.mul(reset_gate,  h))) + self.bh)
        h = torch.mul((1 - update_gate), h) + torch.mul(update_gate, candidate_act_vector)
        y = (self.Wy @ h) + self.by
        return y, h


model = GRUCell(input_size=price_tensor.shape[0], hidden_size=hidden_size).to(device)
print(model)

loss_fn = nn.MSELoss(reduction='none')

optimizer = optim.RMSprop(model.parameters(), lr=0.00001)


def run_train(sequences):
    model.train()
    h = torch.zeros(hidden_size, device=device)
    for indices in sequences:
        X_window = X_train[:, indices[0]:indices[1]]
        loss = torch.zeros(10, device=device)
        for pairs in [(j, j+1) for j in range(0, X_window.shape[1]-1)]:
            X = X_window[:, pairs[0]]
            y = X_window[:, pairs[1]]
            X, y = X.to(device), y.to(device)
            y_pred, h_state = model(X, h)
            h = h_state
            temp = torch.sum(loss_fn(y_pred, y)[44, 49, 32, 9, 17, 11])
            loss += temp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h = h.detach()
    return h


def run_test(seqn, h):
    loss_fn = nn.MSELoss(reduction='none')
    model.eval()
    test_loss = torch.zeros(hidden_size, device=device)
    with torch.no_grad():
        for indices in seqn:
            X_window = X_test[:, indices[0]:indices[1]]
            for pairs in [(j, j+1) for j in range(0, X_window.shape[1]-1)]:
                X = X_window[:, pairs[0]]
                y = X_window[:, pairs[1]]
                X, y = X.to(device), y.to(device)
                y_pred, h_state = model(X, h)
                h = h_state
                test_loss += loss_fn(y_pred, y)
            h = h.detach()
    test_loss /= 245
    test_loss_copy = test_loss.cpu().numpy()
    tl_series = pd.Series(data=test_loss_copy, index=prices.index)
    print(tl_series)


epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    hidden_state = run_train(sequence_train)
    run_test(sequence_test, hidden_state)
print("Done!")



