
from model import UNIF, Encoder 
from dataset import get_dataset 
from config import get_config, save_config
import torch 
import pdb 
from tqdm import tqdm
from argparse import ArgumentParser
import os
from colorama import Fore
import numpy as np 
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random

def set_seed(seed):
    print("seed: ", seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:

    def __init__(self, config, rank = 0, world_size = 1, ddp = False):
     
        #load the config
        set_seed(101)
        self.config = config
        self.config.ddp = ddp
        self.config.world_size = world_size
        self.rank = rank 

        #load the models
        self.encoder = Encoder().cuda()
        self.model = UNIF().cuda()

        #it ddp then
        if(self.config.ddp):
            self.model = DDP(self.model)
            self.encoder = DDP(self.encoder)
        
        if(self.rank == 0):
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\nTotal Trainable Parameters: {total_params}")

        #get the datasets
        #adjust the batch size
        self.config.train.batch_size = self.config.train.batch_size // world_size 
        self.train_dataset = get_dataset(self.config, self.encoder, 'train')
        self.val_dataset = get_dataset(self.config, self.encoder, 'val')
        
        #the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        
        
        if(rank == 0):
            #the optimizer
            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
            self.init_files()
            #self.plot_lr()

        #the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        self.lr_scheduler = self.get_lr_scheduler()
    
    def plot_lr(self):
        
        scheduler = self.get_lr_scheduler()
        epoch_steps = len(self.train_dataset)
        total_steps = epoch_steps*self.config.epochs
        lr = []
        x = []
        for i in tqdm(range(int(total_steps))):
            if(i % epoch_steps == 0):
                scheduler.step()
            lr.append(self.optimizer.param_groups[0]['lr'])
            x.append(i / epoch_steps)
        #pdb.set_trace()
        plt.plot(x,lr)
        plt.savefig(os.path.join(self.folder_path,'lr.png'))

    def get_lr_scheduler(self):

        return torch.optim.lr_scheduler.StepLR(self.optimizer, int(self.config.epochs/2), 0.5)

    def init_files(self):
        
        #the root folder
        self.folder_path = os.path.join(self.config.root, self.config.experiment_name) 
        os.makedirs(self.folder_path, exist_ok = True)

        #the checkpoint folder
        self.checkpoint_folder = os.path.join(self.folder_path, 'checkpoints')
        os.makedirs(self.checkpoint_folder, exist_ok = True)

        #the loss csv
        self.train_loss = os.path.join(self.folder_path, 'train_loss.csv')
        self.val_loss = os.path.join(self.folder_path, 'val_loss.csv')
        open(self.train_loss, 'w')
        open(self.val_loss, 'w')

        #dump the config file
        save_config(self.config, os.path.join(self.folder_path, 'config.yaml'))

    
    def train_epoch(self):

        #train for one epoch
        losses = {'loss': []}
        
        pbar = tqdm(self.train_dataset, ncols = 120, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        pbar.set_postfix({"loss":100, "rank:": self.rank, "i": 0})
        
        for batch_data in pbar:
            
            #put data on cuda
            self.optimizer.zero_grad()
            batch_data.to('cuda')
            output = self.model(batch_data)
            loss   = output['loss']
            loss.backward()
            self.optimizer.step()
            
            #append the loss
            losses['loss'].append(loss.item())
        
            pbar.set_postfix({"loss":loss.item(),  "rank:": self.rank})
            pbar.update()

        for k in losses:
            losses[k] = sum(losses[k]) / len(losses[k])
        losses['lr'] = self.optimizer.param_groups[0]['lr']
        return losses

    def save_results(self, data, mode = 'train'):

        if(mode == 'train'):
            f = open(self.train_loss, 'a')
        else:
            f = open(self.val_loss, 'a')
        for k in data:
            f.write(f"{data[k]},")
        f.write("\n")
    
    def save_model(self):
        
        if(self.config.ddp):
            model = self.model.module
        else:
            model = self.model 
        
        checkpoint = dict()
        checkpoint['model'] = model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(self.checkpoint_folder, f'{self.epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

    def print_results(self, results, mode ='train'):
        if(mode == 'train'):
            print(f"Epoch: {self.epoch} ")
        else:
            print(f"VALIDATION")
        for k in results:
            print(f'{k}: {results[k]}', end = ' |')
        print()
        print()

    
    def train(self):
        
        #train for epochs
        for epoch in range(self.config.epochs):
            
            # train for one epoch
            self.epoch = epoch
            loss = self.train_epoch()
            self.lr_scheduler.step()

            #set the epoch no
            if(self.config.ddp):
                self.train_dataset.sampler.set_epoch(self.epoch)

            with torch.no_grad():
                if(self.rank == 0):

                    #print the results
                    self.print_results(loss)

                    #save the results
                    self.save_results(loss, 'train')

                    if(self.epoch % self.config.save_every == 0):
                        
                        #save the model
                        self.save_model()

                    if(self.epoch % self.config.validate_every == 0):
                        #validate
                        val_loss = self.validate()

                        #save the results
                        self.save_results(val_loss, 'val')

                        #print the results
                        self.print_results(val_loss, 'val')
            
    
    def validate(self):

        #validate for all datasets
        results = {}
        for i, dataset in enumerate(self.val_dataset):
            results_i = self.validate_i(dataset, i)
            results.update(results_i)
        return results
    
    def validate_i(self, dataset, i):
        
        y_trues = []
        y_preds = []
        loss = []
        results = {}

        pbar = tqdm(dataset, ncols = 120, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        for data in pbar:
            #put data on cuda
            data.to('cuda')
            output = self.model(data)
            
            #num_classes
            num_classes = data.num_classes[0].item()
            y_true = output['y_true'].view(-1, num_classes)
            y_true = y_true.cpu().detach().numpy()

            y_pred =  output['y_pred'].view(-1, num_classes)
            y_pred = y_pred.cpu().detach().numpy()

            y_trues.append(y_true)
            y_preds.append(y_pred)
            loss.append(output['loss'].item())

            pbar.update()
        
        y_trues = np.concatenate(y_trues, axis = 0)
        y_preds = np.concatenate(y_preds, axis = 0)

        results[f'loss_{i}'] = sum(loss)/len(loss)

        if(self.config.val_datasets[i]['type_'] == 'classification'):
            accuracy = self.accuracy(y_trues, y_preds)
            results[f'accuracy_{i}'] = accuracy
        else:
            rmse = self.rmse(y_trues, y_preds)
            results[f'rmse_{i}'] = rmse
        return results

    def rmse(self, y_true, y_pred):
        y_pred = y_pred[:,0]
        y_true = y_true[:,0]

        rmse = ((y_pred - y_true)**2).mean()
        return rmse**0.5

    def accuracy(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = -1)
        y_true = np.argmax(y_true, axis = -1)
        accu = (y_pred == y_true).sum() / y_pred.shape[0]
        return accu

def ddp_setup(local_rank):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    #os.environ["MASTER_ADDR"] = "172.20.9.5"
    #os.environ["WORLD_SIZE"] = "2"
    #os.environ["MASTER_PORT"] = "12350"

    init_process_group(backend="nccl")#, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def main_ddp(args = None):
    
    rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    ddp_setup(rank)
    trainer = Trainer(args, global_rank, world_size, True)
    trainer.train()
    destroy_process_group()
    
def main(args):
    # init_process_group(backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    trainer = Trainer(args)
    trainer.train()



def get_parser():
    
    #the parser for the arguments
    parser = ArgumentParser(
                        prog = 'python main.py',
                        description = 'This is main file for transferable structured world model. It either trains mmodel form scratch or produces test results for the test data',
                        epilog = 'thank you!!')

    #there are two tasks ['train', 'test']
    parser.add_argument('--args', nargs='+', required=False,  default = [], 
                        help='arguments for the config file in the form [key value] eg. dataset gravity epoch 200')

    #there are two tasks ['train', 'test']
    parser.add_argument('--ddp',  
                        help='whether to go for ddp', action = 'store_true', default = False)

    return parser

if __name__ == '__main__':

    #parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    
    config = get_config(args.args)
    if(args.ddp):
        main_ddp(config)
    else:
        main(config)

