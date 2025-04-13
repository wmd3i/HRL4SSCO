import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv
import math

aggregation_function = 'mean'
activation_function = nn.ReLU()


class Encoder(nn.Module):
    def __init__(self, node_feature_size=3, embedding_layers_size=128, heads=8, graph_layers_num=3, ff_layers_size=512):
        super(Encoder, self).__init__()
        self.node_feature_size = node_feature_size
        self.embedding_layers_size = embedding_layers_size

        self.heads = heads
        self.graph_output_size = self.embedding_layers_size // self.heads
        self.graph_layers_num = graph_layers_num

        self.ff_layers_size = ff_layers_size

        # initial embedding
        self.embedding_layer = nn.Linear(
            self.node_feature_size, self.embedding_layers_size)

        self.graph_layers = nn.ModuleList()
        for i in range(self.graph_layers_num):
            self.graph_layers.append(
                GATConv(self.embedding_layers_size, self.graph_output_size, heads=self.heads))

        self.ff_layers = nn.ModuleList()
        for i in range(self.graph_layers_num):
            self.ff_layers.append(
                nn.Linear(self.embedding_layers_size, self.ff_layers_size))
            self.ff_layers.append(
                nn.Linear(self.ff_layers_size, self.embedding_layers_size))

        self.batch_norms = nn.ModuleList()
        for i in range(self.graph_layers_num):
            self.batch_norms.append(BatchNorm1d(self.embedding_layers_size))
            self.batch_norms.append(BatchNorm1d(self.embedding_layers_size))

        self.activation = nn.ReLU()

    def forward(self, x_, edge_index):
        x = self.embedding_layer(x_)

        for i in range(self.graph_layers_num):
            y = self.graph_layers[i](x, edge_index)
            x = self.batch_norms[2 * i](x + y)
            y = self.ff_layers[2 * i](x)
            y = self.activation(y)
            y = self.ff_layers[2 * i + 1](y)
            x = self.batch_norms[2 * i + 1](x + y)
        return x


class decoderI(nn.Module):
    def __init__(self, node_feature_size=128, linear_layers_size=[64, 32]):
        super(decoderI, self).__init__()
        self.node_feature_size = node_feature_size
        self.linear_layers_size = linear_layers_size

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(
            nn.Linear(self.node_feature_size + 3, self.linear_layers_size[0]))
        self.linear_layers.append(NoisyLinear(
            self.linear_layers_size[0], self.linear_layers_size[1]))
        self.linear_layers.append(NoisyLinear(self.linear_layers_size[1], 1))

        self.activation = activation_function

    def forward(self, x, action):
        x = x.mean(dim=0)
        x = torch.cat([x, action], 0)
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[-1](x)
        return x


class SimpleModelI(nn.Module):
    def __init__(self, node_feature_size, embed_dim=64, T=3):
        super(SimpleModelI, self).__init__()
        self.node_feature_size = node_feature_size
        self.embed_dim = embed_dim
        self.T = T

        self.q1 = nn.Linear(self.node_feature_size, self.embed_dim)
        self.q2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.q3 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.q4 = nn.Linear(
            self.embed_dim, self.embed_dim)       # for mu_mean

        self.q5 = torch.nn.Linear(1, self.embed_dim)  # for action

        self.p1 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.p2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.p3 = nn.Linear(self.embed_dim, 1)

    def forward(self, xv, action, adj):

        adj_ = adj

        hv = self.q1(xv).clamp(0)

        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(adj_, hv)
            else:
                mu = torch.matmul(adj_, mu)
            mu1 = self.q2(mu).clamp(0)
            mu_cat = torch.cat((hv, mu1), dim=-1)
            mu = self.q3(mu_cat).clamp(0)

        mu_mean = mu.mean(dim=0)
        mu_mean = self.q4(mu_mean).clamp(0)

        action = action.view(-1, 1)
        q5 = self.q5(action).clamp(0).view(-1)

        q_ = torch.cat((mu_mean, q5), dim=-1)

        q_ = self.p1(q_).clamp(0)
        q_ = self.p2(q_).clamp(0)
        q = self.p3(q_)
        return q


# class decoderII(nn.Module):
#     def __init__(self, embedding_size=128, heads=8):
#         super(decoderII, self).__init__()
#         self.embedding_size = embedding_size
#         self.heads = heads

#         self.initial_embedding = nn.Linear(
#             self.embedding_size + 1, self.embedding_size)

#         self.attention_layer1 = AttentionLayer(embedding_size, heads)
#         self.attention_layer2 = AttentionLayer(embedding_size, 1)

#         self.linear_layer = NoisyLinear(1, 1)

#         self.activation = nn.Tanh()

#     def forward(self, x, option):
#         graph_embedding = x.mean(dim=0).unsqueeze(0)
#         option = option.unsqueeze(0)

#         context_embedding = torch.cat([graph_embedding, option], 1)
#         context_embedding = self.initial_embedding(context_embedding)

#         y, _ = self.attention_layer1(context_embedding, x)
#         _, compatibility = self.attention_layer2(y, x)

#         compatibility = compatibility.view(-1, 1)
#         compatibility = self.activation(compatibility)
#         compatibility = (compatibility - compatibility.mean()
#                          ) / torch.std(compatibility)

#         output = self.linear_layer(compatibility)

#         return output


class decoderII(nn.Module):
    def __init__(self, node_feature_size=128, linear_layers_size=[64, 128]):
        super(decoderII, self).__init__()
        self.node_feature_size = node_feature_size
        self.linear_layers_size = linear_layers_size

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(
            nn.Linear(self.node_feature_size + 1, linear_layers_size[0]))
        self.linear_layers.append(NoisyLinear(
            linear_layers_size[0], linear_layers_size[1]))
        self.linear_layers.append(NoisyLinear(linear_layers_size[1], 1))

        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(BatchNorm1d(self.node_feature_size + 1))
        for i in range(len(self.linear_layers_size)):
            self.batch_norms.append(BatchNorm1d(self.linear_layers_size[i]))

        self.null_q = nn.Parameter(torch.Tensor(1, 1))

        self.activation = nn.ReLU()
        self.last_activation = nn.Sigmoid()

        self.init_parameters()

    def init_parameters(self):
        self.null_q.data.uniform_(0, 1)

    def forward(self, x, option):
        option = option.unsqueeze(0)
        option = option.repeat(x.size(0), 1)
        y = torch.cat([x, option], 1)

        for i in range(len(self.linear_layers)):
            y = self.batch_norms[i](y)
            y = self.linear_layers[i](y)
            if i != len(self.linear_layers) - 1:
                y = self.activation(y)

        y = torch.cat([y, self.null_q], 0)
        y = self.last_activation(y)
        y = y.view(-1)
        return y


class NetworkI(nn.Module):
    def __init__(self, node_feature_size=3, embedding_layers_size=128, heads=8, graph_layers_num=3, ff_layers_size=512, linear_layers_size=[64, 32]):
        super(NetworkI, self).__init__()

        self.encoder = Encoder(node_feature_size=node_feature_size, embedding_layers_size=embedding_layers_size,
                               heads=heads, graph_layers_num=graph_layers_num, ff_layers_size=ff_layers_size)
        self.decoder = decoderI(
            node_feature_size=embedding_layers_size, linear_layers_size=linear_layers_size)

    def forward(self, x, action, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x, action)
        return x


class NetworkII(nn.Module):
    def __init__(self, node_feature_size=3, embedding_layers_size=128, heads=8, graph_layers_num=3, ff_layers_size=512, linear_layers_size=[64, 32]):
        super(NetworkII, self).__init__()

        self.encoder = Encoder(node_feature_size=node_feature_size, embedding_layers_size=embedding_layers_size,
                               heads=heads, graph_layers_num=graph_layers_num, ff_layers_size=ff_layers_size)
        self.decoder = decoderII(
            node_feature_size=embedding_layers_size, linear_layers_size=linear_layers_size)
        # self.decoder = decoderII(
        #     embedding_size=embedding_layers_size, heads=heads)

    def forward(self, x, option, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x, option)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, heads, value_dim=None, key_dim=None):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.value_dim = input_dim // heads if value_dim is None else value_dim
        self.key_dim = self.value_dim if key_dim is None else key_dim

        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.W_query = nn.Parameter(torch.Tensor(
            self.heads, self.input_dim, self.key_dim))
        self.W_key = nn.Parameter(torch.Tensor(
            self.heads, self.input_dim, self.key_dim))
        self.W_val = nn.Parameter(torch.Tensor(
            self.heads, self.input_dim, self.value_dim))
        self.W_out = nn.Parameter(torch.Tensor(
            self.heads, self.value_dim, self.input_dim))

        self.init_parameters()

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        Query = torch.matmul(q, self.W_query)
        Key = torch.matmul(h, self.W_key)
        Value = torch.matmul(h, self.W_val)

        compatibility = torch.matmul(
            Query, Key.transpose(1, 2)) * self.norm_factor

        heads = torch.matmul(compatibility, Value)
        output = torch.mm(heads.view(-1, self.value_dim *
                          self.heads), self.W_out.view(-1, self.input_dim))
        return output, compatibility

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


class SimpleModel(nn.Module):
    def __init__(self, node_feature_size):
        super(SimpleModel, self).__init__()
        self.node_feature_size = node_feature_size
        self.c1 = torch.nn.Parameter(torch.Tensor([[1.], [0.], [0.]]))
        self.fc_a = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.fc_p = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, xv, adj, mask=None):
        batch_size = xv.shape[0]
        node_num = xv.shape[1]
        adj_ = adj.unsqueeze(0)
        adj_ = adj_.expand(batch_size, node_num, node_num)

        mu = torch.matmul(adj_, xv)
        c1 = self.c1.expand(batch_size, self.node_feature_size, 1)
        # c1 = c1.view(batch_size, self.node_feature_size, 1)
        q = torch.bmm(mu, c1)
        q_mean = q.mean(dim=1).unsqueeze(1)

        q = self.fc_a(q)
        q_mean = self.fc_p(q_mean)
        q = torch.cat((q, q_mean), dim=1)

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            mask_tensor = mask_tensor.view(batch_size, -1, 1)
            q[mask_tensor == 1] = -99999
        return q


class SimpleGNN(nn.Module):
    def __init__(self, node_feature_size, embed_dim=64, T=1):
        super(SimpleGNN, self).__init__()
        self.node_feature_size = node_feature_size
        self.embed_dim = embed_dim
        self.T = T

        self.q1 = torch.nn.Linear(self.node_feature_size, self.embed_dim)
        self.q2 = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.q3 = torch.nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.fc_a = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        self.fc_p = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, xv, adj, mask=None):
        batch_size, node_num = xv.shape[0], xv.shape[1]

        adj_ = adj.unsqueeze(0)
        adj_ = adj_.expand(batch_size, node_num, node_num)

        hv = self.q1(xv).clamp(0)

        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(adj_, hv)
            else:
                mu = torch.matmul(adj_, mu)
            mu1 = self.q2(mu).clamp(0)
            mu_cat = torch.cat((hv, mu1), dim=-1)
            mu = self.q3(mu_cat).clamp(0)

        mu_mean = mu.mean(dim=1).unsqueeze(1)

        q_a = self.fc_a(mu)
        q_p = self.fc_p(mu_mean)
        q = torch.cat((q_a, q_p), dim=1)

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            mask_tensor = mask_tensor.view(batch_size, -1, 1)
            q[mask_tensor == 1] = -99999

        return q


class S2V_modified(torch.nn.Module):
    def __init__(self, node_feature_size, embed_dim=64, reg_hidden=64, len_pre_pooling=0, len_post_pooling=0, T=4, weighted_edge=True):

        super(S2V_modified, self).__init__()
        self.T = T
        self.node_feature_size = node_feature_size
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.weighted_edge = weighted_edge

        self.mu_1 = torch.nn.Parameter(
            torch.Tensor(self.node_feature_size, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)
        if self.weighted_edge:
            self.mu_3 = torch.nn.Linear(embed_dim, embed_dim, True)
            torch.nn.init.normal_(self.mu_3.weight, mean=0, std=0.01)
            self.mu_4 = torch.nn.Linear(1, embed_dim, True)
            torch.nn.init.normal_(self.mu_4.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(1, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)

        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(
                2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = nn.Linear(self.reg_hidden, 1)
            # self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = nn.Linear(2 * embed_dim, 1)
            # self.q = torch.nn.Linear(2 * embed_dim, 1)
        # torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, option, adj, mask=None):

        batch_size = xv.shape[0]
        node_num = xv.shape[1]

        A = adj.clone()
        if self.weighted_edge:
            W = adj.clone()
            A = (adj > 0).float()
        A = A.unsqueeze(0)
        W = W.unsqueeze(-1)

        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(xv, self.mu_1).clamp(0)
            else:
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)

                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(A, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = (mu_1 + mu_2).clamp(0)

                if self.weighted_edge:
                    mu_4 = self.mu_4(W).clamp(0)
                    mu_4 = mu_4.sum(dim=1).unsqueeze(0)
                    mu_3 = self.mu_3(mu_4)
                    mu = (mu_1 + mu_2 + mu_3).clamp(0)

        mu_mean = mu.mean(dim=1).unsqueeze(1)
        mu_mean = self.q_1(mu_mean).clamp(0)
        # batch_size * (node_num + 1) * embed_dim
        mu_aug = torch.cat((mu, mu_mean), dim=1)
        option = option.reshape(batch_size, -1, 1)
        q_2 = self.q_2(option)
        q_2 = q_2.expand(batch_size, node_num + 1, self.embed_dim)
        # batch_size * (node_num + 1) * (embed_dim * 2)
        q_ = torch.cat((mu_aug, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_ = q_.clamp(0)
            q = self.q(q_)

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            mask_tensor = mask_tensor.view(batch_size, -1, 1)
            q[mask_tensor == 1] = -99999
        return q


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
