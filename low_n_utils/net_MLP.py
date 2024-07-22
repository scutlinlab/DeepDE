import copy
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr
sys.path.append("/home/wangqihan/Low-N-improvement/function")
import config
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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

    def fit(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.NET_LOSS['learning_rate'], weight_decay=config.NET_LOSS['weight_decay'])
        loss_fn = nn.MSELoss(reduction='sum')

        full_dataset = EmbeddedDataset(x, y)
        val_size = int(config.NET_LOSS['val_set_prop'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=config.NET_LOSS['train_batch_size'])
        val_loader = DataLoader(dataset=val_dataset, pin_memory=True, batch_size=config.NET_LOSS['val_batch_size'])

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
                if (iter_ + 1) % config.NET_LOSS['accumulated_iters'] == 0:
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

            criterion = config.NET_LOSS['criterion']
            patience = config.NET_LOSS['patience']     
            history4ending = val_history[criterion]   
            if len(history4ending) > patience:
                prev_perform = history4ending[-patience-1]
                window = history4ending[-patience:]
                if criterion != 'loss':
                    end_training = (prev_perform > max(window) - config.NET_LOSS['converg_thld'])
                else:
                    end_training = (prev_perform < min(window) + config.NET_LOSS['converg_thld'])
                   
            if criterion != 'loss':
                    if history4ending[-1] == max(history4ending):
                        self.best_model = copy.deepcopy(self.state_dict())
            else:
                if history4ending[-1] == min(history4ending):
                    self.best_model = copy.deepcopy(self.state_dict())

            if end_training:
                break
            epoch += 1
        
        self.load_state_dict(self.best_model)
    
    def predict(self, x):
        test_dataset = EmbeddedDataset(x)
        test_loader = DataLoader(test_dataset, batch_size=config.NET_LOSS['test_batch_size'])
        yhats = []
        for x in test_loader:
            yhat = self.forward(x.cuda(0))
            yhats.extend(yhat.data.cpu().numpy())
        return yhats


class MLPNoAttention_Lengthen(MLP):
    def __init__(self):
        super().__init__()
        
        # self.fc1 = nn.Linear(config['seq_len'], 1)
        self.fc2 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])
        self.fc3 = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.act = act_dict[config.NET_MLP['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        x = self.act(x.transpose(-2, -1))
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # predict: [BS]
        return x

class MLPNoAttention_unirep(MLP):
    def __init__(self):
        super().__init__()
        config.NET_MLP['mlp_input_size'] = 1900
        # self.fc1 = nn.Linear(config['seq_len'], 1)
        self.fc2 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])
        self.fc3 = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.act = act_dict[config.NET_MLP['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        x = self.act(x.transpose(-2, -1))
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # predict: [BS]
        return x

class MLPNoAttention_onehot(MLP):
    def __init__(self):
        super().__init__()
        config.NET_MLP['mlp_input_size'] = 4780
        # self.fc1 = nn.Linear(config['seq_len'], 1)
        self.fc2 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])
        self.fc3 = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.act = act_dict[config.NET_MLP['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.to(torch.float32)
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        x = self.act(x.transpose(-2, -1))
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # predict: [BS]
        return x
    
class MLPNoAttention_euni_a(MLP):
    def __init__(self):
        super().__init__()
        config.NET_MLP['mlp_input_size'] = 4781
        # self.fc1 = nn.Linear(config['seq_len'], 1)
        self.fc2 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])
        self.fc3 = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.act = act_dict[config.NET_MLP['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.to(torch.float32)
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        x = self.act(x.transpose(-2, -1))
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # predict: [BS]
        return x

class MLPNoAttention(MLP):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(config.NET_MLP['seq_len'], 1)
        self.fc2 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])
        self.fc3 = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.act = act_dict[config.NET_MLP['mlp_hidden_act']]
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # BS = 1
        # x.T: [BS, D0, L] -> x: [BS, D0, 1]
        # print(x.shape)
        # print(x.transpose(-2, -1).shape)
        x = self.act(self.fc1(x.transpose(-2, -1)))
        # print(x.shape)
        # print(x.transpose(-2, -1).shape)
        # x.T: [BS, 1, D0] -> x: [BS, 1, D1]
        x = self.ln(self.act(self.fc2(x.transpose(-2, -1))))
        # print(x.shape)
        # x: [BS, 1, D1] -> [BS, 1, 1] -> [BS]
        x = self.fc3(x).squeeze(-1).squeeze(-1)
        # print(x.shape)
        # predict: [BS]
        return x

class MLPFixedHiddenReLU(MLP):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.NET_MLP['mlp_input_size'], config.NET_MLP['mlp_hidden_size'])
        self.fc2 = nn.Linear(config.NET_MLP['seq_len'], 1)

        self.output = nn.Linear(config.NET_MLP['mlp_hidden_size'], 1)
        self.ln = nn.LayerNorm(config.NET_MLP['mlp_hidden_size'])

    def forward(self, arrays):
        hidden1 = self.ln(torch.relu(self.fc1(arrays)))
        hidden2 = torch.sigmoid(self.fc2(hidden1.transpose(-2, -1)))
        output = self.output(hidden2.transpose(-2, -1))
        output = output.squeeze(-1).squeeze(-1)
        return output

class MLPAttentionSigmoid_new(nn.Module):
    def __init__(self, input_dim, stacked_dim, hidden_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(4,1)

        self.ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, arrays):
        x1 = self.ln(torch.relu(self.fc1(arrays))) # x1:[length, input_dim] -> [length, hidden_dim]
        # Attention_1
        x2 = F.softmax(self.fc2(x1), dim = 0) # x2:[length, hidden_dim] -> [length, 1]
        hidden_0 = torch.mm(x1.transpose(0, 1), x2) # hidden: [hidden_dim, length] * [length, 1] -> [hidden_dim, 1]
        # Attention_2
        hidden_1 = x1.mean(0)
        hidden_2 = x1.max(0)[0]
        hidden_3 = x1.min(0)[0]

        hidden = torch.stack((hidden_0.transpose(0, 1).squeeze(), hidden_1, hidden_2, hidden_3)) # [4, hidden_dim]
        x3 = torch.relu(self.fc3(hidden.transpose(0,1)))
        output = self.output(x3.transpose(0,1))

        return output

def train_mlp(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPNoAttention().cuda(0)
    model.fit(x_train, y_train)
    return model

def train_mlp_fh(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPFixedHiddenReLU().cuda(0)
    model.fit(x_train, y_train)
    return model

def train_mlp_uni(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPNoAttention_unirep().cuda(0)
    # model = MLPNoAttention().cuda(0)
    model.fit(x_train, y_train)
    return model

def train_mlp_onehot(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPNoAttention_onehot().cuda(0)
    # model = MLPNoAttention().cuda(0)
    model.fit(x_train, y_train)
    return model

def train_mlp_euni_a(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPNoAttention_euni_a().cuda(0)
    # model = MLPNoAttention().cuda(0)
    model.fit(x_train, y_train)
    return model

def train_mlp_Lengthen(seed, x_train, y_train):
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    model = MLPNoAttention_Lengthen().cuda(0)
    model.fit(x_train, y_train)
    return model
