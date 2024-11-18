from yacs.config import CfgNode

default_config = CfgNode({

    'root': '../experiments',
    'experiment_name': 'ind_arxiv_regression_more_layers',
    'lr': 1e-4,
    'epochs': 60,
    'train':{
        'batch_size': 128,
        'num_workers': 0
    },
    'val':{
        'batch_size': 128,
        'num_workers': 0
    },
    
    'train_datasets':[
        {   
            'name': 'arxiv_regression', 
            'level': 'node',
            'type_': 'regression' ,
            'full': True
        }
    ],

    'val_datasets':[
        {
            'name': 'arxiv_regression',
            'level': 'node',
            'type_': 'regression' ,
            'full': True
        }
    ],

    'test_datasets':[
        {
            'name': 'arxiv_regression',
            'level': 'node',
            'type_': 'regression' ,
            'full': True
        }
    ],
    'dataset_roots':{
        'arxiv': '/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/arxiv/data.pt',
        'cora': '/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/cora/data.pt',
        'fb15k': '/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/fb15k/data.pt',
        'wn18rr': '/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/wn18rr/data.pt',
        'arxiv_regression': '/home/scai/phd/aiz238706/scratch/COL870-GNN/project/unif/data/arxiv_regression/data.pt'
    },
    'validate_every':2,
    'save_every':2
})



#return the overriden config
def get_config(extra_args):
    default_config.set_new_allowed(True)
    default_config.merge_from_list(extra_args)
    default_config.extras = extra_args
    return default_config


#dump to file
def save_config(config, file_name):
    config_str = config.dump()
    with open(file_name, "w") as f:
        f.write(config_str)