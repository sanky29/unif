import pdb 
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph
from scipy.sparse import csr_array
import numpy as np 
from torch_geometric.data import Data
import random 
from torch.nn.functional import one_hot



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
            self.get_noi = lambda x: [self.graph.nodes[x].item()]
        elif(self.type_ == 'edge'):
            self.get_noi = lambda x: [self.edge_index[:,self.graph.edges[x]][0].item(), self.edge_index[:,self.graph.edges[x]][1].item()] 
        else:
            pdb.set_trace()

        
        #the get y function
        if(self.type_ == 'node'):
            self.get_label = lambda x: self.graph.y[self.graph.nodes[x]: self.graph.nodes[x] + 1]
        elif(self.type_ == 'edge'):
            self.get_label = lambda x: self.graph.y[self.graph.edges[x]:self.graph.edges[x] + 1] 
        else:
            pdb.set_trace()

    def __getitem__(self, i):
        
        #choose the nois
        nois = self.get_noi(i)

        #sample nodes
        nodes = sample_fixed_hop_size_neighbor(self.adj, nois, self.hop, self.max_nodes)
        
        #make the neighbours
        neighbors = np.r_[nois, nodes]
        
        #get the node features
        x = self.graph.x[neighbors]
        
        #make the edges
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
        subgraph.NOI = torch.tensor([k for k in range(len(nois))])
        subgraph.y = self.get_label(i)
        
        #print('subgraph.y : ', subgraph.y.shape)
        return subgraph



## from a subgraph (whether query or support), generate an NOI subgraph given
##### - prompt node embedding
##### - prompt edge embedding
##### - task level (node/edge/graph)
##### - NOI (node ID/edge ID/None) respectively
def annotate_subgraph(graph, prompt_node_embedding, prompt_edge_embedding, task_level, NOI):
    
    
    num_nodes = graph.num_nodes ## also the index of the (to-be-added) prompt nod
    X = torch.cat((graph.x, prompt_node_embedding.unsqueeze(0)), dim=0)

    #the original indices
    edge_indices = graph.edge_index
    edge_attrs   = graph.edge_attr

    #add NOI - NOI PROMPT NODE edges
    for i in NOI:
        edge_indices = torch.cat([edge_indices] +  [torch.tensor([[i], [num_nodes]]), torch.tensor([[num_nodes], [i]])], dim=1)    
    edge_attrs   = torch.cat([edge_attrs]   +  [prompt_edge_embedding.unsqueeze(0) for i in NOI] + [prompt_edge_embedding.unsqueeze(0) for i in NOI], dim=0)
    
    new_graph = Data(x = X, edge_index=edge_indices, edge_attr=edge_attrs)
    new_graph.y           = graph.y
    new_graph.prompt_node = num_nodes
    return new_graph

### Given query and k support graphs, construct the graph that is to be passed to the model
def combine(query_graph, class_node_embeddings , encoder, query_prompt_edge_embedding):
    ## all graphs already has their's prompt Node appended to them appropriately
    ## support_graphs is a list of graphs
    #### big graph has the following structure : 
    ####### x          = qg.x + class_nodes.x
    ####### edge_index = qg.e + qg-class_nodes
    ####### edge_attr  = qg.a + qg-class_nodes
    ####### y          = qg.y

    ### Step 1 : Add all nodes + 2 (min/max)
    x = torch.cat([query_graph.x]  + [class_node_embeddings], dim=0)

    #get the num classes
    num_classes = class_node_embeddings.shape[0]

    ##### Step 2 : handling edge_index
    min_node_num  =  query_graph.num_nodes

    ## num_nodes_offset is now [vq, vq+v1, vq + v1 + v2, ...., vq + v1 + .. vn]
    edge_indices = query_graph.edge_index

    prompt_nodes = []
    prompt_nodes.append(query_graph.prompt_node)
    for prompt_node in prompt_nodes:
        for c in range(num_classes):
            edge_indices = torch.cat((edge_indices, torch.tensor([[prompt_node], [c + min_node_num]])), dim=1)
            edge_indices = torch.cat((edge_indices, torch.tensor([[c+min_node_num], [prompt_node]])), dim=1)
        

    ###### Handling edge_attr
    edge_attr  = query_graph.edge_attr
    edge_attr = torch.cat([edge_attr] + [query_prompt_edge_embedding.unsqueeze(0) for _ in range(2*num_classes)], dim=0)
 
    #### combining
    big_graph   = Data(x = x, edge_index=edge_indices, edge_attr=edge_attr)
    #big_graph.y = one_hot(query_graph.y, num_classes = num_classes).view(-1)
    big_graph.y = torch.tensor([query_graph.y, 1-query_graph.y])
    big_graph.class_nodes = torch.tensor([0 for i in range(min_node_num)] + [min_node_num + c for c in range(num_classes)])
    big_graph.num_classes = num_classes
    return big_graph


'''
The main dataset of the graph
'''

class UnifRegDatasetFull(Dataset):


    def __init__(self,  root = None, 
                        hops = 2,
                        max_nodes = 100,
                        encoder = None, 
                        mode = 'train', 
                        level = 'node'):
        
        #load a graph dataset
        if(mode == 'val'):
            mode = 'valid'
        self.graph = torch.load(root)
        self.graph.total_labels = self.graph.label_embeddings.shape[0]

        #the type of dataset
        self.type_ = level

        #the encoder
        self.encoder = encoder

        #the subgraph sampler
        self.subgraph = SubGraphDataset(self.graph, hops, max_nodes, level)

        #the class embeddings
        self.class_node_embeddings = self.graph.label_embeddings
        self.mode = mode

        #splitting train, val, test
        self.create_split()

        #select the graph nodes
        if(self.type_ == 'node'):
            self.graph.nodes = self.graph.split[mode]
            self.graph.num_nodes = len(self.graph.nodes)
        elif(self.type_ == 'edge'):
            self.graph.edges = self.graph.split[mode]
            self.graph.num_edges = len(self.graph.edges)
        else:
            self.graph.graphs = self.graph.split[mode]
        

    def create_split(self):
        if(self.type_ == 'node'):
            t = [i for i in range(self.graph.num_nodes)]
        elif(self.type_ == 'edge'):
            t = [i for i in range(self.graph.num_edges)]
        else:
            t = [i for i in range(self.graph.num_graphs)]
        if(self.graph.get('split') is None):
            self.graph.split = dict()
            self.graph.split['train'] = torch.tensor(t[:int(0.8*len(t))])
            self.graph.split['valid'] = torch.tensor(t[int(0.8*len(t)):int(0.9*len(t))])
            self.graph.split['test'] = torch.tensor(t[int(0.9*len(t)):int(len(t))])

    def __len__(self):
        ml = 1
        if(self.mode == 'val'):
            ml = 0.1
        if(self.type_ == 'node'):
            return int(ml*self.graph.num_nodes)
        elif(self.type_ == 'edge'):
            return int(ml*self.graph.num_edges) 
        return int(ml*self.graph.num_graphs)

    def __getitem__(self, i):

        query_subgraph = self.subgraph[i]
        prompt_node_embedding = self.graph.noi_node_embeddings[0,:]
        prompt_edge_embedding = self.graph.prompt_edge_embeddings[0]
        query_prompt_edge_embedding = self.graph.prompt_edge_embeddings[1]
        query_subgraph = annotate_subgraph(query_subgraph, prompt_node_embedding, prompt_edge_embedding, task_level=self.type_, NOI=query_subgraph.NOI)
      
        combined_graph = combine(query_subgraph, self.class_node_embeddings, self.encoder, query_prompt_edge_embedding)

        return combined_graph

if __name__ == '__main__':
    root = "/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/arxiv/data.pt"

    dataset = SubGraphDataset(root)
   