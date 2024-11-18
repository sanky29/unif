import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=1, out_features=768)

    def forward(self, x):
        #print("x.shape to encoder input ", x.shape)
        x = self.l1(x)
        return x
        
import pdb
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_scatter import scatter_softmax

class UNIF(MessagePassing):

    def __init__(self):
        super(UNIF, self).__init__(aggr = 'mean')

        #hidden dimensions
        self.hidden_dim = 768

        #hidden layers
        self.K = 6

        #the mlps
        self.mlps = []
        for i in range(self.K):
            mlpi = nn.Sequential(
                #nn.ReLU(),
                #nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.mlps.append(mlpi)
        self.mlps = nn.ModuleList(self.mlps)

        #the edge matrices 
        self.edge_proj = []
        for i in range(self.K):
            edge_proji = nn.Sequential(
                #nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias = False),
            )
            self.edge_proj.append(edge_proji)
        self.edge_proj = nn.ModuleList(self.edge_proj)

        #the relu for message
        self.relu = nn.ReLU()

        #the keys and queries matrices
        self.w_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_query = nn.Linear(self.hidden_dim, self.hidden_dim)

        #it is a classigication loss
        self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1)
        )

        #loss
        self.loss = nn.MSELoss()

        #the softmax
        self.softmax = nn.Softmax(dim = -1)

    
    def forward(self, data):
        '''
        Args:
            data.x: [NUM_NODES x NODE_ATTR]
            data.edge_attr: [NUM_NODES x EDGE_ATTR]
            data.y: list of NUM_GRAPHS elements
            data.class_nodes: 
        '''
        #x: [NUM_NODEX x HIDDEN_DIM]
        x = data.x
        
        #edges: [NUM_EDGES x HIDDEN_DIM]
        edges = data.edge_attr
        
        #the edge_index: [2 x NUM_EDGES]
        edge_index = data.edge_index

        #the hidden state list
        hidden_states = [x]
        
        #will run the messages
        for i in range(self.K):
            
            #edges: [NUM_EDGES x HIDDEN_DIM]
            edges_i = self.edge_proj[i](edges)

            #messages: [NUM_NODES x HIDDEN_DIM]
            messages = self.propagate(edge_index, x=x, edge_attr=edges_i)
            
            #update the hidden dimension
            x = self.mlps[i](x + messages)

            #append the x to hidden state
            hidden_states.append(x)
        
        #concatenate the hidden sates
        #x: [NUM_NODES x 1 x hidden_dim]
        data.class_nodes = data.class_nodes.nonzero()[:,0]
        queries = hidden_states[0][:,None,:]
        queries = queries[data.class_nodes]
        queries = self.w_query(queries)

        #the keys
        #keys: [NUM_NODES x k x hidden_dim]
        values = torch.stack(hidden_states[1:], dim = 1)
        values = values[data.class_nodes]
        keys = self.w_key(values)
        
        #attention scores
        #attn: [B x 1 x k]
        attn = torch.bmm(queries, keys.permute(0,2,1))
        attn = self.softmax(attn / (self.hidden_dim ** 0.5))

        #x: [TOTAL_CLASS_NODES x 768]
        x = torch.bmm(attn, values)[:,0,:]
        
        #x: [TOTAL_CLASS_NODES]
        x = self.w_proj(x)[...,0]

        #make the group ids
        group_ids = [i for i, k in enumerate(data.num_classes) for j in range(k)]
        #group_softmax: [TOTAL_CLASS_NODES]
        y_pred = scatter_softmax(x, torch.tensor(group_ids).cuda())
      
        #the loss is the kl divergence
        #y_pred = x
        loss = self.loss(y_pred, data.y)
        loss = loss.sum() / data.num_graphs

        #loss = loss.mean()

        #the results
        results ={
            'y_pred': y_pred,
            'y_true': data.y,
            'loss':loss
        }
        return results

    #the messages
    def message(self, x_j, edge_attr):
        '''
        Args:
            x_j: [NUM_EDGES x HIDDEN_DIM]
            edge_attr: [NUM_EDGES x HIDDEN_DIM]
        '''
        message = self.relu(x_j + edge_attr)
        return message
    
    def loss(self, y_pred, y_true):
        '''
        Args:
            y_pred: [TOTAL_CLASS_NODES]
            y_true: [TOTAL_CLASS_NODES]
        '''
        loss = -1*y_true*torch.log(y_pred+1e-10) 
        #loss -= (1 - y_true)*torch.log(1 - y_pred+1e-10)
        return loss

