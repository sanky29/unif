from .full_reg import UnifRegDatasetFull
from .full_class import UnifClassDatasetFull
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pdb 
__all__ = ['get_dataset']

def get_datasets_(config, encoder, mode):
    
    config['encoder'] = encoder
    config['mode'] = mode
    del config['name']
    if(config['type_'] == 'classification'):
        del config['type_']
        if(config['full']):
            del config['full']
            return UnifClassDatasetFull(**config)
    
    elif(config['type_'] == 'regression'):
        del config['type_']
        if(config['full']):
            del config['full']
            return UnifRegDatasetFull(**config)


def get_dataset(config, encoder, mode = 'train'):

    datasets = []
    for config_i in config.train_datasets:
        config_i = dict(config_i)
        config_i['root'] = config.dataset_roots[config_i['name']]
        dataset_i = get_datasets_(config_i, encoder, mode)
        datasets.append(dataset_i)
    
    

    batch_size = config[mode].batch_size
    num_workers = config[mode].num_workers
    shuffle = (mode == 'train')
    #if ddp
    if(mode == 'train'):
        dataset = ConcatDataset(datasets)
        if(config.ddp):
            #now just concate the datasets
            dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        sampler=DistributedSampler(dataset), 
                        num_workers=num_workers, 
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=(num_workers > 0))
        else:
            dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle,  
                        num_workers=num_workers)
    else:
        dataloader = []
        for dataset_i in datasets:
            dataloader.append( DataLoader(dataset_i, 
                        batch_size=batch_size, 
                        shuffle=shuffle,  
                        num_workers=num_workers))
    return dataloader