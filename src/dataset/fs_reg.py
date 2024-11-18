import pdb 
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph
from scipy.sparse import csr_array
import numpy as np 
from torch_geometric.data import Data
import random 

def sample_fixed_hop_size_neighbor(adj_mat: object, root: object, hop: object, max_nodes_per_hop: object = 500) -> object:
    visited = np.array(root)
    fringe  = np.array(root)
    nodes   = np.array([])
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) > max_nodes_per_hop:
            fringe = np.random.choice(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = np.concatenate([nodes, fringe])
        # dist_list+=[dist+1]*len(fringe)
    nodes = nodes.astype(int)
    return nodes


class SubGraphDataset(Dataset):

    def __init__(self, graph, hop = 2, max_nodes = 100, type_ = 'node'):

        #load the grap
        self.graph = graph
        self.hop = hop 
        self.max_nodes = max_nodes
        self.type_ = type_
        
        #k hop neighbourhood
        self.edge_index = self.graph.edge_index

        #get the adj list in sparse settings
        self.adj = csr_array((torch.ones(len(self.edge_index[0])), (self.edge_index[0], self.edge_index[1]),),
                                 shape=(self.graph.num_nodes, self.graph.num_nodes), )

        #the noi function
        if(self.type_ == 'node'):
            self.get_noi = lambda x: [x]
        elif(self.type_ == 'edge'):
            self.get_noi = lambda x: [self.edge_index[:,x][0].item(), self.edge_index[:,x][1].item()] 
        else:
            pdb.set_trace()

    def __getitem__(self, i):
        
        #choose the nois
        nois = self.get_noi(i)

        #sample nodes
        nodes = sample_fixed_hop_size_neighbor(self.adj, nois, self.hop, self.max_nodes)
        nodes = np.array(list(set(nodes).union(set(nois))))
        
        #get the node features
        x = self.graph.x[nodes]
        
        #get the edges
        neighbors = np.r_[i, nodes]
        edges = self.adj[neighbors, :][:, neighbors].tocoo()
        edge_index = torch.stack(
            [torch.tensor(edges.row, dtype=torch.long), torch.tensor(edges.col, dtype=torch.long), ])
        
        #get the edge attributes
        if self.graph.get('edge_type') is not None:

            uniqStep = torch.max(self.graph.edge_index)
            edge_ids = self.graph.edge_index[0, :] * uniqStep + self.graph.edge_index[1, :]
            slided_ids = edge_index[0, :] * uniqStep + edge_index[1, :]
            sliced_set = torch.tensor(slided_ids)
            edge_mask = torch.isin(edge_ids, sliced_set)

            edge_type_list = self.graph['edge_type'][edge_mask]
            edge_attr_subgraph = self.edge_embeddings[edge_type_list]

        else:
            # Extract edge attributes for the subgraph
            edge_attr_subgraph = self.graph.edge_embeddings[0].unsqueeze(0).expand(edge_index.shape[1], -1)


        #edge_attr_subgraph = self.graph.edge_embeddings

        # Create the subgraph as a Data object
        subgraph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_subgraph)
        if self.type_ == "node":
            subgraph.NOI = torch.tensor(np.where(nodes == i)[0])
        elif self.type_ == "edge":
            subgraph.NOI = torch.where((edge_index[0] == nois[0]) & (edge_index[1] == nois[1]))[0]
        else:
            subgraph.NOI = None
        subgraph.y = self.graph.y[i:i+1]
        #print('subgraph.y : ', subgraph.y.shape)

        return subgraph



## from a subgraph (whether query or support), generate an NOI subgraph given
##### - prompt node embedding
##### - prompt edge embedding
##### - task level (node/edge/graph)
##### - NOI (node ID/edge ID/None) respectively
def annotate_subgraph(graph, prompt_node_embedding, prompt_edge_embedding, task_level, NOI):
    num_nodes = graph.num_nodes ## also the index of the (to-be-added) prompt node
    
    X = torch.cat((graph.x, prompt_node_embedding.unsqueeze(0)), dim=0)
    edge_indices = graph.edge_index
    edge_attrs   = graph.edge_attr
    if task_level == "node":
        noi_node = NOI
        ## append bi-directional edge between noi and prompt_node
        edge_indices = torch.cat([edge_indices] +  [torch.tensor([[noi_node], [num_nodes]]), torch.tensor([[num_nodes], [noi_node]])], dim=1)
        edge_attrs   = torch.cat([edge_attrs]   +  [prompt_edge_embedding.unsqueeze(0), prompt_edge_embedding.unsqueeze(0)], dim=0)

    elif task_level == "edge":
        noi_nodes = edge_indices[:, NOI]
        noi_a = noi_nodes[0]
        noi_b = noi_nodes[1]
        edge_indices = torch.cat([edge_indices] +  [torch.tensor([[noi_a], [num_nodes]]), torch.tensor([[num_nodes], [noi_a]])], dim=1)
        edge_indices = torch.cat([edge_indices] +  [torch.tensor([[noi_b], [num_nodes]]), torch.tensor([[num_nodes], [noi_b]])], dim=1)
        edge_attrs   = torch.cat([edge_attrs] + [prompt_edge_embedding.unsqueeze(0) for _ in range(4)], dim=0)

    else:
        ## graph task
        additional_edges = [torch.tensor([[i], [num_nodes]]) for i in range(num_nodes)] + [torch.tensor([[num_nodes], [i]]) for i in range(num_nodes)]
        edge_indices = torch.cat([edge_indices] + additional_edges, dim=1)
        edge_attrs   = torch.cat([edge_attrs] + [prompt_edge_embedding.unsqueeze(0) for _ in range(2*num_nodes)], dim=0)

    new_graph = Data(x = X, edge_index=edge_indices, edge_attr=edge_attrs)
    new_graph.y           = graph.y
    new_graph.prompt_node = num_nodes
    return new_graph

### Given query and k support graphs, construct the graph that is to be passed to the model
def combine(query_graph, support_graphs, class_node_embeddings , encoder, query_prompt_edge_embedding):
    ## all graphs already has their's prompt Node appended to them appropriately
    ## support_graphs is a list of graphs


    #### big graph has the following structure : 
    ####### x          = qg.x + sg.x + min_.x + max_.x
    ####### edge_index = qg.e + [sg.e] + [sg-min + sg-max] + qg-min + qg-max
    ####### edge_attr  = qg.a + [sg.a] + [sg-min + sg-max] + qg-min + qg-max
    ####### y          = qg.y

    ### Step 1 : Add all nodes + 2 (min/max)
    x = torch.cat([query_graph.x] + [support_graph.x for support_graph in support_graphs] + [class_node_embeddings], dim=0)


    ##### Step 2 : handling edge_index
    num_nodes    = [g.num_nodes for g in ([query_graph] + support_graphs)]
    min_node_num = sum(num_nodes)
    
    num_nodes_offset = []
    curr = 0
    for v in num_nodes:
        curr += v
        num_nodes_offset.append(curr)
    ## num_nodes_offset is now [vq, vq+v1, vq + v1 + v2, ...., vq + v1 + .. vn]
    edge_indices = [query_graph.edge_index] + [(g.edge_index + num_nodes_offset[i]) for i, g in enumerate(support_graphs)]
    edge_indices = torch.cat(edge_indices, dim=1)

    prompt_nodes = []
    for i, sg in enumerate(support_graphs):
        prompt_nodes.append(sg.prompt_node + num_nodes_offset[i])
    prompt_nodes.append(query_graph.prompt_node)

    
    for prompt_node in prompt_nodes:
        for c in range(class_node_embeddings.shape[0]):
            edge_indices = torch.cat((edge_indices, torch.tensor([[prompt_node], [c + min_node_num]])), dim=1)
            edge_indices = torch.cat((edge_indices, torch.tensor([[c+min_node_num], [prompt_node]])), dim=1)
        

    ###### Handling edge_attr
    edge_attr  = torch.cat([query_graph.edge_attr] + [support_graph.edge_attr for support_graph in support_graphs], dim=0)
    #support_vals_min = [encoder(1-sg.y).unsqueeze(0) for sg in support_graphs]
    #sg.y: [NUM_CLASSES] or [2]
    support_vals = [encoder(torch.tensor([sg.y, 1 - sg.y]).unsqueeze(-1)) for sg in support_graphs]

    support_vals_attrs = []
    for i in range(len(support_graphs)):
        support_vals_attrs.append(support_vals[i])
        support_vals_attrs.append(support_vals[i])
       
    edge_attr = torch.cat([edge_attr] + support_vals_attrs + [query_prompt_edge_embedding.unsqueeze(0) for _ in range(2*class_node_embeddings.shape[0])], dim=0)

    #### combining
    big_graph   = Data(x = x, edge_index=edge_indices, edge_attr=edge_attr)
    big_graph.y = torch.tensor([query_graph.y, 1-query_graph.y])
    big_graph.class_nodes = torch.tensor([0 for i in range(min_node_num)] + [min_node_num + c for c in range(class_node_embeddings.shape[0])])
    big_graph.num_classes = 2
    return big_graph


'''
The main dataset of the graph
'''

class UnifRegDatasetKshot(Dataset):


    def __init__(self,  root = None, 
                        k_shots_max = 5, 
                        k_shots_min = 3,
                        hops = 2,
                        max_nodes = 100,
                        encoder = None, 
                        mode = 'train', 
                        type_ = 'node'):
        
        #load a graph dataset
        self.graph = torch.load(root)

        #splitting train, val, test    
        self.subgraph = SubGraphDataset(self.graph, hops, max_nodes, type_)
        
        #the few shot examples
        self.k_shots_max = k_shots_max
        self.k_shots_min = k_shots_min

        #the type of dataset
        self.type_ = type_

        #the encoder
        self.encoder = encoder


        self.min_class_node_prompt = self.graph.label_embeddings[0]
        self.max_class_node_prompt = self.graph.label_embeddings[1]

    def __len__(self):

        if(self.type_ == 'node'):
            return self.graph.num_nodes
        elif(self.type_ == 'edge'):
            return self.graph.num_edges 
        return self.graph.num_graphs

    def __getitem__(self, i):

        query_subgraph = self.subgraph[i]
        support_subgraphs = [self.subgraph[random.randint(0, len(self)-1)] for _ in range(self.k_shots_max)]

        prompt_node_embedding = self.graph.noi_node_embeddings[0,:]
        prompt_edge_embedding = self.graph.prompt_edge_embeddings[0]
        query_prompt_edge_embedding = self.graph.prompt_edge_embeddings[1]

        query_subgraph = annotate_subgraph(query_subgraph, prompt_node_embedding, prompt_edge_embedding, task_level=self.type_, NOI=query_subgraph.NOI)
        support_subgraphs = [annotate_subgraph(s, prompt_node_embedding, prompt_edge_embedding, task_level=self.type_, NOI=s.NOI) for s in support_subgraphs]

        combined_graph = combine(query_subgraph, support_subgraphs, torch.stack([self.min_class_node_prompt, self.max_class_node_prompt],dim = 0), self.encoder, query_prompt_edge_embedding)

        return combined_graph

if __name__ == '__main__':
    root = "/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/arxiv/data.pt"

    dataset = SubGraphDataset(root)
    pdb.set_trace()