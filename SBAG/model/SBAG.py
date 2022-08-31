import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.SBAG_trainer import SBAG_trainer
import torch_geometric.utils as utils
import torch.nn.functional as F
import numpy as np
from .GAT import GATConv
#from torch_geometric.nn import GCNConv

class Attention(nn.Module):
    def __init__(self, input, hidden):
        super(Attention, self).__init__()
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.W = nn.Parameter(torch.FloatTensor(input, hidden))
        self.init()

    def init(self):
        init.xavier_normal_(self.lin1.weight)
        init.xavier_normal_(self.lin2.weight)
        init.xavier_normal_(self.W)

    def forward(self, K, V, mask = None):
        concat = torch.einsum("bhd,dd,bd->bhd", V, self.W, K)
        fc1 = self.activation(self.lin1(concat))
        score = self.lin2(fc1)

        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        attn_value = (alpha * V).sum(dim=1)
        return attn_value

class SBAG(SBAG_trainer):

    def __init__(self, config):
        super(SBAG, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']

        self.A_us = config['A_us']
        self.A_uu = config['A_uu']
        embeding_size = config['embeding_size']
        dropout_rate = config['dropout']
        alpha = 0.4

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.user_embedding = nn.Embedding(config['A_us'].shape[0], embeding_size, padding_idx=0)
        self.source_embedding = nn.Embedding(config['A_us'].shape[1], embeding_size)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])

        self.Wcm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).cuda()

        self.W1 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W2 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W3 = nn.Parameter(torch.FloatTensor(embeding_size, embeding_size))
        self.linear_fu = nn.Linear(400, 1)

        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.leakyrelu = nn.LeakyReLU()

        self.fc_out = nn.Sequential(
            nn.Linear(300 + 2*embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        )
        

        self.linear = nn.Linear(15, 15).to(self.device)
        checkpoint = torch.load("./model/bd_model.pkl")
        self.linear.load_state_dict({k.replace('linear.',''):v for k,v in checkpoint["prop_cell"].items()})
        self.cls = nn.Linear(15,2).to(self.device)
        init.xavier_normal_(self.cls.weight)

        self.gnn1 = GATConv(embeding_size, out_channels=8, dropout=dropout_rate, heads=8, negative_slope=alpha)
        self.gnn2 = GATConv(64, embeding_size, dropout=dropout_rate, concat=False, negative_slope=alpha)

        self.attention = Attention(embeding_size, embeding_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.source_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.W3)
        init.xavier_normal_(self.linear_fu.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def publisher_encoder(self, X_user, X_user_id, user_node_feat):

        A_us = self.A_us[X_user_id.cpu(), :].todense()
        A_us = torch.FloatTensor(A_us).cuda()
        M = self.source_embedding.weight

        user_node_cred = self.cls(user_node_feat).softmax(dim=1)[:,1]
    

        user_node_cred = user_node_cred.unsqueeze(1).expand(user_node_cred.shape[0],M.shape[1])

        m_hat = torch.einsum("bs,sd,dd->bd", A_us, M, self.W3) + user_node_cred
        m_hat = self.relu(m_hat)
        
        U_hat = m_hat + X_user

        return U_hat

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row]
        x_j = x[col]

        xi_weight = self.cls(x_i).softmax(dim=1)
        xi_weight = xi_weight[:,1]
        xj_weight = self.cls(x_j).softmax(dim=1)
        xj_weight = xj_weight[:,1]
        edge_weight = (xi_weight+xj_weight)/2
        return edge_weight

    def retweet_user_encoder(self, X_ruser, X_ruser_id, user_node_feat, test_flag): 

        node_init_feat =  self.user_embedding.weight
 
        edge_index = self.A_uu.edge_index.cuda()
        edge_weight = self.edge_infer(user_node_feat, edge_index)

        node_rep1 = self.gnn1(node_init_feat, edge_index, edge_weight)
        node_rep1 = self.dropout(node_rep1)
        
        graph_output, (edge_index, alpha) = self.gnn2(node_rep1, edge_index, edge_weight, return_attention_weights=True)

        return graph_output[X_ruser_id]

    def text_representation(self, X_word):

        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim=1)
        features = self.dropout(features)
        return features

    def forward(self, X_source_wid, X_source_id, X_user_id, X_ruser_id, test_flag = False): 

        X_word = self.word_embedding(X_source_wid)

        user_node = self.A_uu.node_feat.cuda()
        user_node_feat = torch.tanh(self.linear(user_node))

        X_user = self.user_embedding(X_user_id)
        X_ruser = self.user_embedding(X_ruser_id)


        X_text = self.text_representation(X_word)

        user_rep = self.publisher_encoder(X_user, X_user_id, user_node_feat[X_user_id])
 
        
        r_user_rep = self.retweet_user_encoder(X_ruser, X_ruser_id, user_node_feat, test_flag)
        mask = ((X_ruser_id != 0) == 0)

        ru_count = torch.where(mask==False, 1, 0).sum(dim=1)

        ru_rep = self.attention(user_rep, r_user_rep, mask)

        tweet_rep = torch.cat([X_text, user_rep, ru_rep], dim=-1)

        Xt_logit = self.fc_out(tweet_rep)

        return Xt_logit

class SBAG_weibo(SBAG_trainer):

    def __init__(self, config):
        super(SBAG_weibo, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']

        self.A_us = config['A_us']
        self.A_uu = config['A_uu']
        embeding_size = config['embeding_size']
        dropout_rate = config['dropout']
        alpha = 0.4

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.user_embedding = nn.Embedding(config['A_us'].shape[0], embeding_size, padding_idx=0)
        self.source_embedding = nn.Embedding(config['A_us'].shape[1], embeding_size)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])

        self.Wcm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).cuda()

        self.W1 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W2 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W3 = nn.Parameter(torch.FloatTensor(embeding_size, embeding_size))
        self.linear_fu = nn.Linear(400, 1)

        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        self.fc_out = nn.Sequential(
            nn.Linear(300 + 2*embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        )

        self.linear = nn.Linear(10, 10).to(self.device)
        checkpoint = torch.load("./model/bd_model10.pkl")

        self.linear.load_state_dict({k.replace('linear.',''):v for k,v in checkpoint["prop_cell"].items()})
        self.cls = nn.Linear(10,2).to(self.device)

        init.xavier_normal_(self.cls.weight)

        self.gnn1 = GATConv(embeding_size, out_channels=8, dropout=dropout_rate, heads=8, negative_slope=alpha)
        self.gnn2 = GATConv(64, embeding_size, dropout=dropout_rate, concat=False, negative_slope=alpha)

        self.attention = Attention(embeding_size, embeding_size)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.source_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.W3)
        init.xavier_normal_(self.linear_fu.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)


    def publisher_encoder(self, X_user, X_user_id, user_node_feat):

        A_us = self.A_us[X_user_id.cpu(), :].todense()
        A_us = torch.FloatTensor(A_us).cuda()
        M = self.source_embedding.weight

        user_node_cred = self.cls(user_node_feat).softmax(dim=1)[:,1]

        user_node_cred = user_node_cred.unsqueeze(1).expand(user_node_cred.shape[0],M.shape[1])

        m_hat = torch.einsum("bs,sd,dd->bd", A_us, M, self.W3) + user_node_cred
        m_hat = self.relu(m_hat)
        
        U_hat = m_hat + X_user

        return U_hat

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row]
        x_j = x[col]
       
        xi_weight = self.cls(x_i).softmax(dim=1)
        xi_weight = xi_weight[:,1]
        xj_weight = self.cls(x_j).softmax(dim=1)
        xj_weight = xj_weight[:,1]

        edge_weight = (xi_weight+xj_weight)/2

        return edge_weight

    def retweet_user_encoder(self, X_ruser, X_ruser_id, user_node_feat): 
        
        node_init_feat =  self.user_embedding.weight

        edge_index = self.A_uu.edge_index.cuda()
        edge_weight = self.edge_infer(user_node_feat, edge_index)
 
        node_rep1 = self.gnn1(node_init_feat, edge_index, edge_weight)
        node_rep1 = self.dropout(node_rep1)

        graph_output = self.gnn2(node_rep1, edge_index, edge_weight)

        return graph_output[X_ruser_id]

    def text_representation(self, X_word):

        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim=1)
        features = self.dropout(features)
        return features

    def forward(self, X_source_wid, X_source_id, X_user_id, X_ruser_id, test_flag = False): 

        X_word = self.word_embedding(X_source_wid)


        user_node = self.A_uu.node_feat.cuda()
        user_node_feat = torch.tanh(self.linear(user_node))

        X_user = self.user_embedding(X_user_id)
        X_ruser = self.user_embedding(X_ruser_id)

        X_text = self.text_representation(X_word)

        user_rep = self.publisher_encoder(X_user, X_user_id, user_node_feat[X_user_id])

        
        r_user_rep = self.retweet_user_encoder(X_ruser, X_ruser_id, user_node_feat)
   
        mask = ((X_ruser_id != 0) == 0)

        ru_rep = self.attention(user_rep, r_user_rep, mask)

        tweet_rep = torch.cat([X_text, user_rep, ru_rep], dim=-1)
        Xt_logit = self.fc_out(tweet_rep)

        return Xt_logit