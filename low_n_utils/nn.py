import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr

def swish(x):
    return x * torch.sigmoid(x)

act_dict = {'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'gelu': F.gelu,
            'swish': swish}


class EmbeddedDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        if y is not None:
            assert len(x) == len(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]


class MLP(nn.Module):

    def fit(self, x, y, config):
        optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        loss_fn = nn.MSELoss(reduction='sum')

        full_dataset = EmbeddedDataset(x, y)
        val_size = int(config['val_set_prop'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=config['train_batch_size'])
        val_loader = DataLoader(dataset=val_dataset, pin_memory=True, batch_size=config['val_batch_size'])

        train_loss_history = []
        val_history = {'loss': [], 'r': [], 'rho': []}
        end_training = False
        epoch = 0
        while True:
            for iter_, (train_x, train_y) in enumerate(train_loader):
                train_yhat = self.forward(train_x.cuda(0))
                train_loss = loss_fn(train_yhat, train_y.float().cuda(0))
                train_loss_history.append(float(train_loss))
                train_loss.backward()
                if (iter_ + 1) % config['accumulated_iters'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            val_loss = 0
            val_count = 0
            val_yhats = []
            val_ys = []
            for val_x, val_y in val_loader:
                val_count += val_x.shape[0]
                val_yhat = self.forward(val_x.cuda(0))
                val_loss += float(loss_fn(val_yhat, val_y.float().cuda(0)))
                val_yhats.extend(val_yhat.data.cpu().numpy())
                val_ys.extend(val_y.data.cpu().numpy())

            val_loss = val_loss / val_count
            val_r = pearsonr(val_yhats, val_ys)[0]
            val_rho = spearmanr(val_yhats, val_ys)[0]
            val_history['loss'].append(val_loss)
            val_history['r'].append(val_r)
            val_history['rho'].append(val_rho)
            print(f'epoch {epoch}: val_loss = {round(val_loss, 4)}, r = {round(val_r, 4)}, '
                  f'rho = {round(val_rho, 4)}')

            criterion = config['criterion']
            patience = config['patience']     
            history4ending = val_history[criterion]   
            if len(history4ending) > patience:
                prev_perform = history4ending[-patience-1]
                window = history4ending[-patience:]
                if criterion != 'loss':
                    end_training = (prev_perform > max(window) - config['converg_thld'])
                    if history4ending[-1] == max(history4ending):
                        self.best_model = copy.deepcopy(self.state_dict())
                else:
                    end_training = (prev_perform < min(window) + config['converg_thld'])
                    if history4ending[-1] == min(history4ending):
                        self.best_model = copy.deepcopy(self.state_dict())

            if end_training:
                break
            epoch += 1
        
        self.load_state_dict(self.best_model)
    
    def predict(self, x, config):
        test_dataset = EmbeddedDataset(x)
        test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'])
        yhats = []
        for x in test_loader:
            yhat = self.forward(x.cuda(0))
            yhats.extend(yhat.data.cpu().numpy())
        return yhats


class MLPNoAttention(MLP):
    def __init__(self, config):
        super().__init__()
        
        self.fc1 = nn.Linear(config['seq_len'], 1)
        self.fc2 = nn.Linear(config['mlp_input_size'], config['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config['mlp_hidden_size'])
        self.fc3 = nn.Linear(config['mlp_hidden_size'], 1)
        self.act = act_dict[config['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        x = self.act(self.fc1(x.transpose(-2, -1)))
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # predict: [BS]
        return x
    

def train_mlp(x_train, y_train, config):
    model = MLPNoAttention(config).cuda(0)
    model.fit(x_train, y_train, config)
    return model
