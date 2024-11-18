from sentence_transformers import SentenceTransformer, util
import torch
import sys 
import os
import pdb 
from tqdm import tqdm 
import pandas as pd 

def main(input_folder, output_folder):  

    #load the model
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1').cuda()
    
    #make the directory
    os.makedirs(output_folder, exist_ok = True)

    #load the data
    data = torch.load(os.path.join(input_folder, 'cora.pt'))

    #process the text
    node_embeddings = []
    for xi in tqdm(data.raw_text):
        text_i = "feature node. paper title and abstract: " + xi
        node_embeddings.append(model.encode(text_i))
    data.x = torch.tensor(node_embeddings)

    #the y 
    data.y = data.node_year[:,0]
    data.nodes = (data.y >= 2000).nonzero()[:,0]
    y_min = 2000
    y_max = data.y.max()
    data.y = (data.y - y_min)/(y_max - y_min)
    data.y = torch.clamp(data.y, 0, 1)

    #save the labels
    label_descriptions = pd.read_csv(os.path.join(input_folder, 'categories.csv')).values
    
    #dump the data
    label_text = ['prompt node. publication year and closness to minimum 1971',
                'prompt node. publication year and closness to maximum 2020']
    

    #generate the label embeddings
    label_embeddings = []
    for li in label_text:
        label_embeddings.append(model.encode(li))
    
    #the label description
    data.label_embeddings = torch.tensor(label_embeddings)
   
    
    #generate the edge embeddings
    edge_text = [
        "feature edge. connected papers are cited together by other papers."
    ]
    edge_embeddings = []
    for li in edge_text:
        edge_embeddings.append(model.encode(li))
    data.edge_embeddings = torch.tensor(edge_embeddings)
    
    #the noi node embeddings
    noi_node_text = [
        "prompt node. node regression on the paper's publication year"
    ]
    noi_node_embeddings = []
    for li in noi_node_text:
        noi_node_embeddings.append(model.encode(li))
    data.noi_node_embeddings = torch.tensor(noi_node_embeddings)
    
    #the noi-class, noi-noi_sugbraph embeddings
    prompt_edge_text = ["prompt edge",
                        "prompt edge. edge for query graph that is our target"]
    prompt_edge_embeddings = []
    for li in prompt_edge_text:
        prompt_edge_embeddings.append(model.encode(li))
    data.prompt_edge_embeddings = torch.tensor(prompt_edge_embeddings)
    
    #dump 
    path = os.path.join(output_folder, 'data.pt')
    torch.save(data, path)

if __name__ == '__main__':

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_folder, output_folder)