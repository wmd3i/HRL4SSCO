import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from graph_encoder import *


# class OptionNetwork(nn.Module):

#     def __init__(self,
#                  node_dim=3,
#                  option_dim=3,
#                  context_dim=2,
#                  embedding_dim=128,
#                  n_encode_layers=3,
#                  normalization='batch',
#                  n_heads=8):
#         super(OptionNetwork, self).__init__()

#         self.node_dim = node_dim
#         self.embedding_dim = embedding_dim
#         self.n_encode_layers = n_encode_layers
#         self.n_heads = n_heads

#         self.init_embed = nn.Linear(node_dim, embedding_dim)

#         self.encoder = GraphAttentionEncoder(
#             n_heads=n_heads,
#             embed_dim=embedding_dim,
#             n_layers=self.n_encode_layers,
#             normalization=normalization,
#         )

#         self.option_embed = nn.Linear(
#             option_dim, option_dim * embedding_dim)
#         # 2 for current and start node
#         self.context_embed = nn.Linear(
#             context_dim, context_dim * embedding_dim)

#         self.decoder = nn.Sequential(
#             nn.Linear(embedding_dim * (1 + context_dim +
#                       option_dim), self.embedding_dim),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim, self.embedding_dim),
#             nn.ReLU(),
#             nn.Linear(self.embedding_dim, 1),
#         )

#         self.init_parameters()

#     def forward(self, x, option, context):
#         """
#         Parameters:
#             x: node features, [batch_size, num_nodes, node_dim]
#             option: current option, [batch_size, option_dim]
#             context: (current node, start node) [batch_size, context_dim]
#         """
#         _, graph_embeddings = self.encoder(self.init_embed(x))
#         option_embedding = self.option_embed(option)
#         context_embedding = self.context_embed(context)
#         feature = torch.cat(
#                 [graph_embeddings, context_embedding, option_embedding], dim=-1)
#         q_value = self.decoder(feature)
#         return q_value

#     def init_parameters(self):

#         for param in self.parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)

class OptionNetwork(nn.Module):

    def __init__(self,
                 node_dim=3,
                 option_dim=3,
                 context_dim=2,
                 embedding_dim=64,
                 T=3):
        super(OptionNetwork, self).__init__()

        self.node_dim = node_dim
        self.option_dim = option_dim
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.T = T

        self.W_K = nn.Linear(2, embedding_dim)
        self.W_Q = nn.Linear(2, embedding_dim)
        self.W_V = nn.Linear(node_dim - 2, embedding_dim)

        self.option_embed = nn.Parameter(
            torch.randn(option_dim, embedding_dim))

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * (1 + context_dim +
                      option_dim), embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        self.init_parameters()

    def forward(self, x, option, context):
        """
#         Parameters:
#             x: node features, [batch_size, num_nodes, node_dim]
#             option: current option, [batch_size, option_dim]
#             context: (current node, start node) [batch_size, context_dim] 
#         """
        loc, p = x[:, :, :2], x[:, :, 2].unsqueeze(-1)
        # q = self.W_Q(loc)
        # k = self.W_K(loc)
        q = loc
        k = loc
        v = self.W_V(p)
        attention_scores = cal_attention_scores(q, k)
        for t in range(self.T):
            if t == 0:
                node_embedding = torch.bmm(attention_scores, v)
            else:
                node_embedding = torch.bmm(attention_scores, node_embedding)
        graph_embedding = torch.mean(node_embedding, dim=1)

        # option_embed = self.option_embed.unsqueeze(0)
        option_embedding = option.unsqueeze(-1) * self.option_embed
        option_embedding = option_embedding.clamp(0).view(x.size(0), -1)

        context = context.unsqueeze(-1).expand(-1, -1,
                                               self.embedding_dim).long()
        context_embedding = torch.gather(node_embedding, 1, context)
        context_embedding = context_embedding.view(x.size(0), -1).clamp(0)
        feature = torch.cat(
            [graph_embedding, context_embedding, option_embedding], dim=-1)

        q_value = self.fc(feature)
        return q_value

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


class ActionNetwork(nn.Module):

    def __init__(self,
                 node_dim=3,
                 context_dim=2,
                 embedding_dim=64,
                 T=1):
        super(ActionNetwork, self).__init__()

        self.node_dim = node_dim
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.T = T

        self.W_K = nn.Linear(2, embedding_dim)
        self.W_Q = nn.Linear(2, embedding_dim)
        self.W_V = nn.Linear(node_dim - 2, embedding_dim)

        self.W2_K = nn.Linear(embedding_dim, embedding_dim)
        self.W2_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W2_V = nn.Linear(embedding_dim, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

        self.fc_null = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        self.init_parameters()

    def forward(self, x, context, mask=None):
        """
        Parameters:
            x: node features, [batch_size, num_nodes, node_dim]
            context: (current node, start node) [batch_size, context_dim] 
            mask: one-hot for invalid nodes, [batch_size, num_nodes + 1], bool
        Returns:
            q_value: [batch_size, num_nodes, 1]
        """
        loc, p = x[:, :, :2], x[:, :, 2].unsqueeze(-1)
        # q = self.W_Q(loc)
        # k = self.W_K(loc)
        q = loc / loc.norm(dim=-1, keepdim=True)
        k = loc / loc.norm(dim=-1, keepdim=True)
        v = self.W_V(p)
        attention_scores = cal_attention_scores(q, k)
        for t in range(self.T):
            if t == 0:
                node_embedding = torch.bmm(attention_scores, v)
            else:
                node_embedding = torch.bmm(attention_scores, node_embedding)
        context = context.unsqueeze(-1).expand(-1, -1,
                                               self.embedding_dim).long()
        # [batch_size, context_dim, embedding_dim]
        # context_embedding = torch.gather(node_embedding, 1, context)
        # q2 = self.W2_Q(node_embedding)
        # k2 = self.W2_K(context_embedding)
        # v2 = self.W2_V(context_embedding)
        # print(q2.shape, k2.shape, v2.shape)
        # attention_scores = cal_attention_scores(
        #     q2, k2)
        # q_value = torch.bmm(attention_scores, v2)

        q_value = self.fc(node_embedding)
        current_node = context[:, 0, :].unsqueeze(1)
        q_value_null = node_embedding.gather(1, current_node)
        q_value_null = self.fc_null(q_value_null)
        q_value = torch.cat([q_value, q_value_null], dim=1)

        if mask is not None:
            mask = mask.view(x.size(0), -1, 1)
            q_value = torch.where(
                mask, torch.full_like(q_value, -float('inf')), q_value)
        return q_value

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


# class ActionNetwork(nn.Module):

#     def __init__(self,
#                  node_dim=3,
#                  context_dim=2,
#                  embedding_dim=128,
#                  n_encode_layers=3,
#                  normalization='batch',
#                  n_heads=8):
#         super(ActionNetwork, self).__init__()

#         self.node_dim = node_dim
#         self.embedding_dim = embedding_dim
#         self.n_encode_layers = n_encode_layers
#         self.n_heads = n_heads

#         self.init_embed = nn.Linear(node_dim, embedding_dim)

#         self.encoder = GraphAttentionEncoder(
#             n_heads=n_heads,
#             embed_dim=embedding_dim,
#             n_layers=self.n_encode_layers,
#             normalization=normalization,
#         )

#         # 2 for current and start node
#         self.context_embed = nn.Linear(
#             context_dim, context_dim * embedding_dim)

#         self.W_Q = nn.Linear(embedding_dim * (1 + context_dim), embedding_dim)
#         self.W_K = nn.Linear(embedding_dim, embedding_dim)
#         self.W_V = nn.Linear(embedding_dim, embedding_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(1, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, 1)
#         )

#         self.fc_null = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, 1)
#         )

#         self.init_parameters()

#     def forward(self, x, context, mask=None):
#         """
#         Parameters:
#             x: node features, [batch_size, num_nodes, node_dim]
#             option: current option, [batch_size, option_dim]
#             context: (current node, start node) [batch_size, context_dim]
#             mask: one-hot for invalid nodes, [batch_size, num_nodes], bool

#         Returns:
#             q_value: [batch_size, num_nodes, 1]
#         """

#         batch_size = x.size(0)
#         node_embeddings, graph_embeddings = self.encoder(self.init_embed(x))
#         context_embedding = self.context_embed(context)

#         q = self.W_Q(torch.cat([graph_embeddings, context_embedding], dim=-1))
#         k = self.W_K(node_embeddings)
#         # v = self.W_V(node_embeddings)

#         q = q.unsqueeze(1)
#         u = torch.bmm(k, q.transpose(1, 2)) / math.sqrt(self.embedding_dim)
#         q_value = self.fc(u)
#         if mask is not None:
#             mask = mask.view(batch_size, -1, 1)
#             print(mask.shape)
#             print(q_value.shape)
#             q_value = torch.where(
#                 mask, torch.full_like(q_value, -float('inf')), q_value)

#         q_value_null = self.fc_null(q)  # for null action
#         q_value = torch.cat([q_value, q_value_null], dim=1)
#         return q_value

#     def init_parameters(self):

#         for param in self.parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)


def cal_attention_scores(q, k):
    '''
    Parameters:
        q: [batch_size, num_queries, embedding_dim] 
        k: [batch_size, num_nodes, embedding_dim]
    Returns:
        attention_scores: [batch_size, num_queries, num_nodes]
    '''
    dot_product = torch.matmul(q, k.transpose(1, 2))
    d = q.shape[-1]
    scaled_dot_product = dot_product / \
        torch.sqrt(torch.tensor(d, dtype=torch.float32))
    attention_scores = F.softmax(scaled_dot_product, dim=-1)
    return attention_scores
