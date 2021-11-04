import numpy as np
import torch
import torch.nn as nn

from . import layers

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer()

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

    def save_network(self, save_pth):
        torch.save(self.state_dict(), save_pth)

    def load_network(self, load_pth):
        state_dict = torch.load(load_pth)
        # remove parameter keys that are not consistent for training and gen repr
        state_dict.pop('proj.inputs.title.emb.weight', None)
        state_dict.pop('scorer.bias', None)
        self.load_state_dict(state_dict, strict=False)
