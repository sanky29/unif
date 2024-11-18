import json
import os.path as osp

import numpy as np
import torch
import torch_geometric as pyg
from sentence_transformers import SentenceTransformer, util
import sys 
from tqdm import tqdm 
import pdb 
import os
def gen_entities(name, input_dir):
    if name == "WN18RR":
        entity2id = {}
        entity_lst = []
        text_lst = []
        with open(osp.join(input_dir, "entity2text.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split("\t")
                entity_lst.append(tmp[0])
                text_lst.append(tmp[1])

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    elif name == "FB15K237":
        entity_lst = []
        text_lst = []
        with open(osp.join(input_dir, "entity2wikidata.json"), "r") as f:
            data = json.load(f)

        for k in tqdm(data):
            # print(data[k])
            entity_lst.append(k)
            text_lst.append("entity names: " + data[k]["label"] + ", entity alternatives: " + ", ".join(
                data[k]["alternatives"]) + ". entity descriptions:" + data[k]["description"] if data[k][
                                                                                                    "description"] is
                                                                                                not None else "None")

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    else:
        raise NotImplementedError("Dataset " + name + " is not implemented.")
    return entity_lst, text_lst, entity2id


def read_knowledge_graph(files, name, input_dir):
    entity_lst, text_lst, entity2id = gen_entities(name, input_dir)
    relation2id = {}

    converted_triplets = {}
    rel_list = []
    rel = len(relation2id)

    for file_type, file_path in files.items():

        edges = []
        edge_types = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]
        unknown_entity = 0
        for triplet in file_data:
            if triplet[0] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[0])
                entity2id[triplet[0]] = len(entity2id)
                unknown_entity += 1
            if triplet[2] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[2])
                entity2id[triplet[2]] = len(entity2id)
                unknown_entity += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel_list.append(triplet[1])
                rel += 1

            edges.append([entity2id[triplet[0]], entity2id[triplet[2]], ])
            edge_types.append(relation2id[triplet[1]])
        print(unknown_entity)
        converted_triplets[file_type] = [edges, edge_types]

    #load the model
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1').cuda()
    

    data = pyg.data.data.Data(x=torch.zeros([len(text_lst), 1]),
        edge_index=torch.tensor(converted_triplets["train"][0]).T,
        edge_types=torch.tensor(converted_triplets["train"][1]), )
    data.y = data.edge_types
    
    #get the node embeddings
    node_text = ["feature node. entity and entity description: " + ent for ent in text_lst]
    node_embeddings = []
    for xi in tqdm(node_text):
        node_embeddings.append(model.encode(xi))
    data.x = torch.tensor(node_embeddings)

    #get the edge embeddings
    edge_text = ["feature edge. relation between two entities. " + relation for relation in rel_list] + [
        "feature edge. relation between two entities. the inverse relation of " + relation for relation in rel_list]
    edge_embeddings = []
    for li in edge_text:
        edge_embeddings.append(model.encode(li))
    data.edge_embeddings = torch.tensor(edge_embeddings)
    
    #label embeddings
    label_text = ["prompt node. relation between two entities. " + relation for relation in rel_list]
    label_embeddings = []
    for li in label_text:
        label_embeddings.append(model.encode(li))
    data.label_embeddings = torch.tensor(label_embeddings)
    
    # # different prompt edge feature for source and target node.
    prompt_edge_text = ["prompt edge.", "prompt edge connected with source node.", "prompt edge connected with target node.",
                        "prompt edge. edge for query graph that is our prediction target.",
                        "prompt edge. edge for support graph that is an example."]
    #the noi-class, noi-noi_sugbraph embeddings
    prompt_edge_text = ["prompt edge",
                    "prompt edge. edge for query graph that is our target"]
    prompt_edge_embeddings = []
    for li in prompt_edge_text:
        prompt_edge_embeddings.append(model.encode(li))
    data.prompt_edge_embeddings = torch.tensor(prompt_edge_embeddings)
    
    #noi node embeddings
    noi_node_text = ["prompt node. relation type prediction between the connected entities.", ]
    noi_node_embeddings = []
    for li in noi_node_text:
        noi_node_embeddings.append(model.encode(li))
    data.noi_node_embeddings = torch.tensor(noi_node_embeddings)
    

    return data


def main(input_path, output_path):
    
    cur_path = input_path
    names = ["train", "valid", "test"]
    name_dict = {n: osp.join(cur_path, n + ".txt") for n in names}
    data = read_knowledge_graph(name_dict, 'WN18RR', input_path)
    path = os.path.join(output_path, 'data.pt')
    torch.save(data, path)

if __name__ == '__main__':

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)
    