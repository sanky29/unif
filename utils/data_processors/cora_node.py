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

 

    #save the labels
    label_descriptions = pd.read_csv(os.path.join(input_folder, 'categories.csv')).values
    
    #dump the data
    label_text = [
        "prompt node. literature category and description: "
        + desc[0]
        + "."
        + desc[1][0]
        for desc in label_descriptions
    ]

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
        "prompt node. node classification on the paper's category"
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