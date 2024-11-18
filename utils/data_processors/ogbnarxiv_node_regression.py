from sentence_transformers import SentenceTransformer, util
import torch
import sys 
import os
import pdb 
from tqdm import tqdm 
import pandas as pd 
from ogb.nodeproppred import PygNodePropPredDataset

def get_node_feature(path):
    # Node feature process
    nodeidx2paperid = pd.read_csv(os.path.join(path, "nodeidx2paperid.csv.gz"), index_col="node idx")
    titleabs_url = ("https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv")
    titleabs = pd.read_csv(titleabs_url, sep="\t", names=["paper id", "title", "abstract"], index_col="paper id", )

    titleabs = nodeidx2paperid.join(titleabs, on="paper id")
    text = ("feature node. paper title and abstract: " + titleabs["title"] + ". " + titleabs["abstract"])
    node_text_lst = text.values

    return node_text_lst


def get_taxonomy(path):
    # read categories and description file
    f = open(os.path.join(path, "arxiv_CS_categories.txt"), "r").readlines()

    state = 0
    result = {"id": [], "name": [], "description": []}

    for line in f:
        if state == 0:
            assert line.strip().startswith("cs.")
            category = ("arxiv " + " ".join(line.strip().split(" ")[0].split(".")).lower())  # e. g. cs lo
            name = line.strip()[7:-1]  # e. g. Logic in CS
            result["id"].append(category)
            result["name"].append(name)
            state = 1
            continue
        elif state == 1:
            description = line.strip()
            result["description"].append(description)
            state = 2
            continue
        elif state == 2:
            state = 0
            continue

    arxiv_cs_taxonomy = pd.DataFrame(result)

    return arxiv_cs_taxonomy


def get_pd_feature(path):
    arxiv_cs_taxonomy = get_taxonomy(path)
    mapping_file = os.path.join(path, "labelidx2arxivcategeory.csv.gz")
    arxiv_categ_vals = pd.merge(pd.read_csv(mapping_file), arxiv_cs_taxonomy, left_on="arxiv category", right_on="id", )
    return arxiv_categ_vals


def get_label_feature(path):
    arxiv_categ_vals = get_pd_feature(path)
    text = ("prompt node. literature category and description: " + arxiv_categ_vals["name"] + ". " + arxiv_categ_vals[
        "description"])
    label_text_lst = text.values

    return label_text_lst


def get_logic_feature(path):
    arxiv_categ_vals = get_pd_feature(path)
    or_labeled_text = []
    not_and_labeled_text = []
    for i in range(len(arxiv_categ_vals)):
        for j in range(len(arxiv_categ_vals)):
            c1 = arxiv_categ_vals.iloc[i]
            c2 = arxiv_categ_vals.iloc[j]
            txt = "prompt node. literature category and description: not " + c1["name"] + ". " + c1[
                "description"] + " and not " + c2["name"] + ". " + c2["description"]
            not_and_labeled_text.append(txt)
            txt = "prompt node. literature category and description: either " + c1["name"] + ". " + c1[
                "description"] + " or " + c2["name"] + ". " + c2["description"]
            or_labeled_text.append(txt)
    return or_labeled_text + not_and_labeled_text


def get_data(dset):
    pyg_data = PygNodePropPredDataset(name="ogbn-arxiv", root=dset.data_dir)
    pyg_data.data.split = pyg_data.get_idx_split()
    cur_path = os.path.dirname(__file__)
    feat_node_texts = get_node_feature(cur_path).tolist()
    class_node_texts = get_label_feature(cur_path).tolist()
    logic_node_texts = get_logic_feature(cur_path)
    feat_edge_texts = ["feature edge. connected papers are cited together by other papers."]
    noi_node_texts = ["prompt node. node classification of literature category"]
    prompt_edge_texts = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(class_node_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                   "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(class_node_texts))],
                                   "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]},
                       "logic_e2e": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                     "class_node_text_feat": ["class_node_text_feat",
                                                              torch.arange(len(class_node_texts),
                                                                           len(class_node_texts) + len(
                                                                               logic_node_texts))],
                                     "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    return ([pyg_data.data], [feat_node_texts, feat_edge_texts, noi_node_texts, class_node_texts + logic_node_texts,
        prompt_edge_texts, ], prompt_text_map,)


def main(input_folder, output_folder):  
    
    #load the model
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1').cuda()
    
    #make the directory
    os.makedirs(output_folder, exist_ok = True)

    #load the data
    pyg_data = PygNodePropPredDataset(name="ogbn-arxiv", root=input_folder)
    pyg_data.data.split = pyg_data.get_idx_split()
    data = pyg_data.data

    #normalize the ys
    data.y = data.node_year[:,0]
    data.nodes = (data.y >= 2000).nonzero()[:,0]
    y_min = 2000
    y_max = data.y.max()
    data.y = (data.y - y_min)/(y_max - y_min)
    data.y = torch.clamp(data.y, 0, 1)

    #process the text
    feat_node_texts = get_node_feature(input_folder).tolist()
    data.raw_text = feat_node_texts
    node_embeddings = []
    for xi in tqdm(data.raw_text):
        text_i = "feature node. paper title and abstract: " + xi
        node_embeddings.append(model.encode(text_i))
    data.x = torch.tensor(node_embeddings)
    
    
    #the class labels
    class_node_texts = ['prompt node. publication year and closness to minimum 1971',
                        'prompt node. publication year and closness to maximum 2020']
    label_embeddings = []
    for li in class_node_texts:
        label_embeddings.append(model.encode(li))
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