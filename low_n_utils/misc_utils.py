
from sklearn import metrics
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def spearman(a,b):
    return stats.spearmanr(a, b).correlation


def classify(a, b, thlda, thldb):
    pred = a
    tgt = b

    tp = fp = tn = fn = 1e-9
    pred_bin = []
    real_bin = []
    for i in range(len(tgt)):
        pred_i = pred[i]
        tgt_i = tgt[i]
        if pred_i > thlda and tgt_i > thldb:
            tp += 1
        if pred_i < thlda and tgt_i > thldb:
            fn += 1
        if pred_i > thlda and tgt_i < thldb:
            fp += 1
        if pred_i < thlda and tgt_i < thldb:
            tn += 1

        if pred_i >= thlda:
            pred_bin.append(1)
        else:
            pred_bin.append(0)
        if tgt_i >= thldb:
            real_bin.append(1)
        else:
            real_bin.append(0)
    print(tp, tn ,fp ,fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    acc_t = tp / (tp + fp)
    acc_f = tn / (tn + fn)
    recall_t = tp / (tp + fn)
    recall_f = tn / (tn + fp)
    fpr, tpr, _ = metrics.roc_curve(real_bin, pred_bin)
    auc = metrics.auc(fpr, tpr)
    return acc, auc
    # return {'acc': acc, 'acc_t': acc_t, 'acc_f': acc_f, 
    #         'recall_t': recall_t, 'recall_f': recall_f, 'auc': auc}


class Knnonehot(Dataset):
    """Extract distance window arrays"""
    def __init__(self, knr_list):
        self.knr_list = knr_list
        #self.file_list = os.listdir(distance_window_path)

    def __len__(self):
        return len(self.knr_list)

    def __getitem__(self, idx):
        #filename = self.file_list[idx]
        arrays = self.knr_list[idx]
        arrays = arrays[:,:15,:9]
        # arrays = np.concatenate((arrays[:, :, :9], arrays[:, :, 10:]), axis=2)
        #print(arrays.shape)
        arrays = arrays.reshape(arrays.shape[0], arrays.shape[1]*arrays.shape[2])
        #filename = filename[:-4]
        # mix_arrays = np.concatenate((arrays[:-1], arrays[1:]), 1)
        # torsions = np.load(os.path.join(torsions_path, filename))
        return arrays

dic = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9, 'I': 10,
       'K': 11, 'M': 12, 'P': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}


def swish_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class Semilabel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, feature_dim=256, output_dim=20):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim//2)
        print("!!!", input_dim, 9, self.input)
        self.ln1 = nn.LayerNorm(hidden_dim//2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln_forward = nn.LayerNorm(hidden_dim)
        self.ln_backward = nn.LayerNorm(hidden_dim)

        self.hidden_forward = nn.Linear(hidden_dim, len(dic))
        self.hidden_backward = nn.Linear(hidden_dim, len(dic))

        # self.predict = nn.Linear(hidden_dim, len(dic))
        self.dropout = nn.Dropout(0.1)

    def forward(self, arrays):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = hidden_states.squeeze(1)
        # hidden_states_forward = swish_fn(self.ln_forward(hidden_states[:, :384]))
        # hidden_states_backward = swish_fn(self.ln_backward(hidden_states[:, 384:]))
        # # print(hidden_states_forward.shape, hidden_states_backward.shape)
        # # hidden_states = swish_fn(self.ln3(hidden_states))
        # # hidden_states = swish_fn(self.ln4(self.hidden(hidden_states)))
        # forward = self.dropout(self.hidden_forward(hidden_states_forward))
        # backward = self.dropout(self.hidden_backward(hidden_states_backward))
        # # output = F.softmax(output, dim=-1)
        # # output = self.dropout(output)

        return hidden_states


class PtsRep(nn.Module):
    """
        CHANGED! DO NOT TEST/EMBED MODELS TRAINED BEFORE 210411
        CHANGED CODE:
            skipped_tokens -> skipped_tokens + 1
            x_f -> x_b
        Do not support batch size > 1
    """

    def __init__(self, input_size=135, hidden_size=384, vocab_size=20, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.ln1 = nn.LayerNorm(hidden_size // 2)

        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_b = nn.LayerNorm(hidden_size)

        self.fc_f = nn.Linear(hidden_size, vocab_size)
        self.fc_b = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.hidden_size = hidden_size


    # def forward(self, data):
    def forward(self, x):
        # x = x.squeeze(0).float().to('cuda:0')
        x = x.float().to('cuda:0')   # modify
        # tgt: [BS, L_max]
        # tgt = data['target'].squeeze(0)
        x = swish_fn(self.ln1(self.fc1(x)))
        x = swish_fn(self.ln2(self.fc2(x)))
        # x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)   # modify

        x, _ = self.lstm(x)
        # x = x.squeeze(1)
        x = x.permute(1, 0, 2)   # modify
        # x_f = swish_fn(self.ln_f(x[:, :self.hidden_size]))
        # x_b = swish_fn(self.ln_b(x[:, self.hidden_size:]))

        # x_f = self.dropout(self.fc_f(x_f))
        # x_b = self.dropout(self.fc_b(x_b))

        # l = len(tgt)
        # shift = self.skipped_tokens + 1
        # total_len = (l - shift - 1) * 2 + (l - shift) * 2
        # loss = 0
        # loss += self.loss_fn(x_f[:l - shift], tgt[shift:]) * (l - shift)
        # loss += self.loss_fn(x_f[:l - shift - 1], tgt[shift + 1:]) * (l - shift - 1)
        # loss += self.loss_fn(x_b[shift:], tgt[:l - shift]) * (l - shift)
        # loss += self.loss_fn(x_b[shift + 1:], tgt[:l - shift - 1]) * (l - shift - 1)

        # loss /= total_len
        # return {'loss': loss}
        return x
    
    # def encoder(self, data):
    #     x = data['input'].squeeze(0).float()
    #     x = swish_fn(self.ln1(self.fc1(x)))
    #     x = swish_fn(self.ln2(self.fc2(x)))
    #     x = x.unsqueeze(1)
    #     x, _ = self.lstm(x)
    #     x = x.squeeze(1)
    #     return x