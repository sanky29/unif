import torch
from torch_geometric.nn import GCNConv, GAT
from torch_geometric.data import Data
from regression_edge_enc_dec import Encoder, Decoder


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
def combine(query_graph, support_graphs, min_prompt, max_prompt, encoder, query_prompt_edge_embedding):
    ## all graphs already has their's prompt Node appended to them appropriately
    ## support_graphs is a list of graphs


    #### big graph has the following structure : 
    ####### x          = qg.x + sg.x + min_.x + max_.x
    ####### edge_index = qg.e + [sg.e] + [sg-min + sg-max] + qg-min + qg-max
    ####### edge_attr  = qg.a + [sg.a] + [sg-min + sg-max] + qg-min + qg-max
    ####### y          = qg.y

    ### Step 1 : Add all nodes + 2 (min/max)
    x            = torch.cat([query_graph.x] + [support_graph.x for support_graph in support_graphs] + [min_prompt.unsqueeze(0), max_prompt.unsqueeze(0)], dim=0)


    ##### Step 2 : handling edge_index
    num_nodes    = [g.num_nodes for g in ([query_graph] + support_graphs)]
    min_node_num = sum(num_nodes)
    max_node_num = min_node_num + 1
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
        edge_indices = torch.cat((edge_indices, torch.tensor([[min_node_num], [prompt_node]])), dim=1)
        edge_indices = torch.cat((edge_indices, torch.tensor([[prompt_node], [min_node_num]])), dim=1)
        edge_indices = torch.cat((edge_indices, torch.tensor([[max_node_num], [prompt_node]])), dim=1)
        edge_indices = torch.cat((edge_indices, torch.tensor([[prompt_node], [max_node_num]])), dim=1)
    

    ###### Handling edge_attr
    edge_attr  = torch.cat([query_graph.edge_attr] + [support_graph.edge_attr for support_graph in support_graphs], dim=0)
    support_vals_min = [encoder(1-sg.y).unsqueeze(0) for sg in support_graphs]
    support_vals_max = [encoder(sg.y).unsqueeze(0) for sg in support_graphs]

    support_vals_attrs = []
    for i in range(len(support_graphs)):
        support_vals_attrs.append(support_vals_min[i])
        support_vals_attrs.append(support_vals_min[i])
        support_vals_attrs.append(support_vals_max[i])
        support_vals_attrs.append(support_vals_max[i])

    edge_attr = torch.cat([edge_attr] + support_vals_attrs + [query_prompt_edge_embedding.unsqueeze(0) for _ in range(4)], dim=0)

    #### combining
    big_graph   = Data(x = x, edge_index=edge_indices, edge_attr=edge_attr)
    big_graph.y = query_graph.y

    return big_graph


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        
        self.encoder     = Encoder()
        self.decoder     = Decoder()

        self.v    = torch.nn.Parameter(torch.rand(64)) ## learnable parameter for edge from query NOI to class nodes

    def forward(self, big_graph):
        ## do message passing

        ## Last two edges are va and vb

        va = big_graph.edge_attr[-2]
        vb = big_graph.edge_attr[-1]

        va = self.decoder(va)
        vb = self.encoder(vb)

        return torch.softmax(torch.cat(va, vb))


if __name__ == "__main__":
    print("hellp")
    edge_index_1 = torch.tensor([[0, 1, 1, 2],
                                 [1, 0, 2, 1]], dtype=torch.long)
    edge_attr_1  = torch.tensor([[5,6,7], [5,6,7], [5,6,7], [5,6,7]], dtype=torch.float)
    x_1 = torch.tensor([[-1,-1], [0,0], [1,1]], dtype=torch.float)
    graph_1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_attr_1)
    graph_1.y = 5

    print("Graph 1 with [key, value.shape]: ", graph_1)

    prompt_node_embedding = torch.tensor([69,69])
    prompt_edge_embedding = torch.tensor([8,8,8])


    #graph_appended = annotate_subgraph(graph_1, prompt_node_embedding, prompt_edge_embedding, "graph", None)
    graph_appended = annotate_subgraph(graph_1, prompt_node_embedding, prompt_edge_embedding, "edge", 2)
    # print("Graph appended has [key, val.shape] :", graph_appended)
    # print("Appended graph node attributes : \n", graph_appended.x)
    # print("Appended edge indices : \n", graph_appended.edge_index)
    # print("Appended edge attrs   : \n", graph_appended.edge_attr)

    prompt_node_embedding_2 = torch.tensor([56,56])
    prompt_edge_embedding_2 = torch.tensor([9,9,9])
    graph_appended_2 = annotate_subgraph(graph_1, prompt_node_embedding_2, prompt_edge_embedding_2, "edge", 1)

    min_node_prompt = torch.tensor([999,999])
    max_node_prompt = torch.tensor([1000, 1000])
    query_prompt_edge_embedding = torch.tensor([444, 444, 444])

    def encoder(x):
        return torch.tensor([333, 333, 333])
    g_meta = combine(graph_appended, [graph_appended_2], min_node_prompt, max_node_prompt, encoder, query_prompt_edge_embedding)
    print(g_meta)
    print(g_meta.edge_index)



    
