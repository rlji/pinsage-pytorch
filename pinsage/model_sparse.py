import numpy as np
import torch
import torch.nn as nn
import dgl

from . import layers

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks, item_emb):
        h_item = self.get_repr(blocks, item_emb)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks, item_emb):
        # project features
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)

        # add to the item embedding itself
        h_item = h_item + item_emb(blocks[0].srcdata[dgl.NID].cpu()).to(h_item)
        h_item_dst = h_item_dst + item_emb(blocks[-1].dstdata[dgl.NID].cpu()).to(h_item_dst)

        return h_item_dst + self.sage(blocks, h_item)

    def save_network(self, save_pth):
        torch.save(self.state_dict(), save_pth)

    def load_network(self, load_pth):
        self.load_state_dict(torch.load(load_pth))
