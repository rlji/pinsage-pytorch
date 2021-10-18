import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader

from dgl.backend import pytorch as F
from dgl import convert, transform
from dgl.sampling import random_walk, select_topk
from dgl import utils

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size, prob=None):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size
        self.prob = prob

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype],
                prob=self.prob,
            )[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers, random_walk_prob=None):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            PinSAGESampler(g, item_type, user_type, random_walk_length, random_walk_prob,
                           random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    #print(old_frontier)
                    #print(frontier)
                    #print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        examples = [getattr(textset[i], field_name) for i in node_ids]

        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + '__len'] = lengths

def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks


"""PinSAGE sampler & related functions and classes"""
"""copy and refactor from dgl.sampling.PinSAGESampler dgl-0.6.1"""


class RandomWalkNeighborSampler(object):
    """PinSage-like neighbor sampler extended to any heterogeneous graphs.

    Given a heterogeneous graph and a list of nodes, this callable will generate a homogeneous
    graph where the neighbors of each given node are the most commonly visited nodes of the
    same type by multiple random walks starting from that given node.  Each random walk consists
    of multiple metapath-based traversals, with a probability of termination after each traversal.

    The edges of the returned homogeneous graph will connect to the given nodes from their most
    commonly visited nodes, with a feature indicating the number of visits.

    The metapath must have the same beginning and ending node type to make the algorithm work.

    This is a generalization of PinSAGE sampler which only works on bidirectional bipartite
    graphs.

    Parameters
    ----------
    G : DGLGraph
        The graph.  It must be on CPU.
    num_traversals : int
        The maximum number of metapath-based traversals for a single random walk.

        Usually considered a hyperparameter.
    termination_prob : float
        Termination probability after each metapath-based traversal.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each given node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors (or most commonly visited nodes) to select for each given node.
    metapath : list[str] or list[tuple[str, str, str]], optional
        The metapath.

        If not given, DGL assumes that the graph is homogeneous and the metapath consists
        of one step over the single edge type.
    weight_column : str, default "weights"
        The name of the edge feature to be stored on the returned graph with the number of
        visits.

    Examples
    --------
    See examples in :any:`PinSAGESampler`.
    """
    def __init__(self, G, num_traversals, prob, termination_prob,
                 num_random_walks, num_neighbors, metapath=None, weight_column='weights'):
        assert G.device == F.cpu(), "Graph must be on CPU."
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals
        self.prob = prob

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError('Metapath must be specified if the graph is homogeneous.')
            metapath = [G.canonical_etypes[0]]
        start_ntype = G.to_canonical_etype(metapath[0])[0]
        end_ntype = G.to_canonical_etype(metapath[-1])[-1]
        if start_ntype != end_ntype:
            raise ValueError('The metapath must start and end at the same node type.')
        self.ntype = start_ntype

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath * num_traversals
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[self.metapath_hops::self.metapath_hops] = termination_prob
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        """
        Parameters
        ----------
        seed_nodes : Tensor
            A tensor of given node IDs of node type ``ntype`` to generate neighbors from.  The
            node type ``ntype`` is the beginning and ending node type of the given metapath.

            It must be on CPU and have the same dtype as the ID type of the graph.

        Returns
        -------
        g : DGLGraph
            A homogeneous graph constructed by selecting neighbors for each given node according
            to the algorithm above.  The returned graph is on CPU.
        """
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, _ = random_walk(
            self.G, seed_nodes, metapath=self.full_metapath, prob=self.prob, restart_prob=self.restart_prob)
        src = F.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        dst = F.repeat(paths[:, 0], self.num_traversals, 0)

        src_mask = (src != -1)
        src = F.boolean_mask(src, src_mask)
        dst = F.boolean_mask(dst, src_mask)

        # count the number of visits and pick the K-most frequent neighbors for each node
        neighbor_graph = convert.heterograph(
            {(self.ntype, '_E', self.ntype): (src, dst)},
            {self.ntype: self.G.number_of_nodes(self.ntype)}
        )
        neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, self.num_neighbors, self.weight_column)
        selected_counts = F.gather_row(counts, neighbor_graph.edata[dgl.EID])
        neighbor_graph.edata[self.weight_column] = selected_counts

        return neighbor_graph


class PinSAGESampler(RandomWalkNeighborSampler):
    """PinSAGE-like neighbor sampler.

    This callable works on a bidirectional bipartite graph with edge types
    ``(ntype, fwtype, other_type)`` and ``(other_type, bwtype, ntype)`` (where ``ntype``,
    ``fwtype``, ``bwtype`` and ``other_type`` could be arbitrary type names).  It will generate
    a homogeneous graph of node type ``ntype`` where the neighbors of each given node are the
    most commonly visited nodes of the same type by multiple random walks starting from that
    given node.  Each random walk consists of multiple metapath-based traversals, with a
    probability of termination after each traversal.  The metapath is always ``[fwtype, bwtype]``,
    walking from node type ``ntype`` to node type ``other_type`` then back to ``ntype``.

    The edges of the returned homogeneous graph will connect to the given nodes from their most
    commonly visited nodes, with a feature indicating the number of visits.

    Parameters
    ----------
    G : DGLGraph
        The bidirectional bipartite graph.

        The graph should only have two node types: ``ntype`` and ``other_type``.
        The graph should only have two edge types, one connecting from ``ntype`` to
        ``other_type``, and another connecting from ``other_type`` to ``ntype``.
    ntype : str
        The node type for which the graph would be constructed on.
    other_type : str
        The other node type.
    num_traversals : int
        The maximum number of metapath-based traversals for a single random walk.

        Usually considered a hyperparameter.
    termination_prob : int
        Termination probability after each metapath-based traversal.

        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each given node.

        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors (or most commonly visited nodes) to select for each given node.
    weight_column : str, default "weights"
        The name of the edge feature to be stored on the returned graph with the number of
        visits.

    Examples
    --------
    Generate a random bidirectional bipartite graph with 3000 "A" nodes and 5000 "B" nodes.

    >>> g = scipy.sparse.random(3000, 5000, 0.003)
    >>> G = dgl.heterograph({
    ...     ('A', 'AB', 'B'): g.nonzero(),
    ...     ('B', 'BA', 'A'): g.T.nonzero()})

    Then we create a PinSage neighbor sampler that samples a graph of node type "A".  Each
    node would have (a maximum of) 10 neighbors.

    >>> sampler = dgl.sampling.PinSAGESampler(G, 'A', 'B', 3, 0.5, 200, 10)

    This is how we select the neighbors for node #0, #1 and #2 of type "A" according to
    PinSAGE algorithm:

    >>> seeds = torch.LongTensor([0, 1, 2])
    >>> frontier = sampler(seeds)
    >>> frontier.all_edges(form='uv')
    (tensor([ 230,    0,  802,   47,   50, 1639, 1533,  406, 2110, 2687, 2408, 2823,
                0,  972, 1230, 1658, 2373, 1289, 1745, 2918, 1818, 1951, 1191, 1089,
             1282,  566, 2541, 1505, 1022,  812]),
     tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2]))

    For an end-to-end example of PinSAGE model, including sampling on multiple layers
    and computing with the sampled graphs, please refer to our PinSage example
    in ``examples/pytorch/pinsage``.

    References
    ----------
    Graph Convolutional Neural Networks for Web-Scale Recommender Systems
        Ying et al., 2018, https://arxiv.org/abs/1806.01973
    """
    def __init__(self, G, ntype, other_type, num_traversals, prob, termination_prob,
                 num_random_walks, num_neighbors, weight_column='weights'):
        metagraph = G.metagraph()
        fw_etype = list(metagraph[ntype][other_type])[0]
        bw_etype = list(metagraph[other_type][ntype])[0]
        super().__init__(G, num_traversals, prob,
                         termination_prob, num_random_walks, num_neighbors,
                         metapath=[fw_etype, bw_etype], weight_column=weight_column)
