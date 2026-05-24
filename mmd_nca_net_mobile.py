import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelfAttentiveEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(hidden_size, 20, bias=False)
        self.ws2 = nn.Linear(20, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.attention_hops = 1

    def forward(self, outp):
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, 20]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, 1]
        alphas = alphas.permute(0, 2, 1)  # [bsz, 1, len]
        alphas = self.softmax(alphas.view(size[0], self.attention_hops, size[1]))  # [bsz, 1, len]
        return torch.bmm(alphas, outp)  # [bsz, 1, nhid]

class A_LSTM(nn.Module):
    def __init__(self, frames_num=90, joints_num=13, dim_num=3):
        super(A_LSTM, self).__init__()
        self.sequence_size = frames_num
        hidden_size = 128  # Each GRU: 128, concatenated to 256
        
        # First layer: input_size=39
        self.gru_forward_l0 = nn.GRU(input_size=joints_num*dim_num, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.gru_backward_l0 = nn.GRU(input_size=joints_num*dim_num, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        # Second layer: input_size=256 (concatenated)
        self.gru_forward_l1 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.gru_backward_l1 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        
        # Register as buffer to ensure proper handling during export
        self.register_buffer('reverse_indices', torch.tensor(list(range(self.sequence_size - 1, -1, -1)), dtype=torch.long))
        
        self.bn1 = nn.BatchNorm1d(self.sequence_size)
        self.selfattention = SelfAttentiveEncoder(hidden_size=hidden_size * 2)  # 256
        self.fc1 = nn.Linear(hidden_size * 2, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(320, 320)
        self.bn3 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(320, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(joints_num*dim_num)

    def forward(self, x):  # [length, bsz, 39]
        length, bsz, feature = x.shape
        x = x.contiguous().view(-1, feature)
        x = self.bn5(x)
        x = x.contiguous().view(length, bsz, feature)

        # Layer 0
        x_forward, _ = self.gru_forward_l0(x)  # [length, bsz, 128]
        x_reversed = x[self.reverse_indices, :, :]
        x_backward, _ = self.gru_backward_l0(x_reversed)  # [length, bsz, 128]
        x_backward = x_backward[self.reverse_indices, :, :]

        # Concatenate for layer 1 input
        x_concat = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        # Layer 1
        x_forward, _ = self.gru_forward_l1(x_concat)
        x_reversed = x_concat[self.reverse_indices, :, :]
        x_backward, _ = self.gru_backward_l1(x_reversed)
        x_backward = x_backward[self.reverse_indices, :, :]

        x = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        # Concatenate
        x = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        x = x.permute(1, 0, 2)  # [bsz, length, 256]
        x = self.bn1(x)
        x = self.selfattention(x).squeeze(1)  # [bsz, 256]
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        l2_norm = torch.norm(x,p=2,dim=1)#bsz
        #print("l2_norm", l2_norm.size())
        return l2_norm

    def gen_embedding(self, x):
        length, bsz, feature = x.shape
        x = x.contiguous().view(-1, feature)
        x = self.bn5(x)
        x = x.contiguous().view(length, bsz, feature)

        # Layer 0
        x_forward, _ = self.gru_forward_l0(x)  # [length, bsz, 128]
        x_reversed = x[self.reverse_indices, :, :]
        x_backward, _ = self.gru_backward_l0(x_reversed)  # [length, bsz, 128]
        x_backward = x_backward[self.reverse_indices, :, :]

        # Concatenate for layer 1 input
        x_concat = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        # Layer 1
        x_forward, _ = self.gru_forward_l1(x_concat)
        x_reversed = x_concat[self.reverse_indices, :, :]
        x_backward, _ = self.gru_backward_l1(x_reversed)
        x_backward = x_backward[self.reverse_indices, :, :]

        x = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        # Concatenate
        x = torch.cat((x_forward, x_backward), dim=-1)  # [length, bsz, 256]

        x = x.permute(1, 0, 2)  # [bsz, length, 256]
        x = self.bn1(x)
        x = self.selfattention(x).squeeze(1)  # [bsz, 256]
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        return x

class MMD_NCA_Mobile_Net(nn.Module):
    def __init__(self, device, frames_num=90, joints_num=13, dim_num=3):
        super(MMD_NCA_Mobile_Net, self).__init__()
        self.sequence_size = frames_num
        self.joints_num=joints_num
        self.dim_num = dim_num
        self.device = device
        self.A_LSTM = A_LSTM(frames_num=frames_num,
                             joints_num=joints_num,
                             dim_num=dim_num)
        self.seq_emb_size = frames_num

    def forward(self, x):
        return self.A_LSTM(x)

    def load_weights(self, weights_path, device):
        self.load_state_dict(torch.load(weights_path))
        self.to(device)
        self.eval()
        self.device = device

    def gen_embedding(self, x):
        tensor = Variable(torch.Tensor(x)).float() \
            .to(self.device).squeeze() \
            .view(-1, self.seq_emb_size, self.joints_num*self.dim_num).permute(1, 0, 2)
        return self.A_LSTM.gen_embedding(tensor)
    
    def calc_embedding(sequence, model, norm=True) :
        if norm :
            sequence = (sequence - sequence.min()) \
                    / (sequence.max() - sequence.min()) # здесь ошибка
        out = model.gen_embedding(sequence)\
                            .cpu().detach().numpy().squeeze()
        return out
    
def calc_embedding(sequence, model, norm=True) :
    if norm :
        sequence = (sequence - sequence.min()) \
                / (sequence.max() - sequence.min()) # здесь ошибка
    out = model.gen_embedding(sequence)\
                        .cpu().detach().numpy().squeeze()
    return out