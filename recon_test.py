import torch

from models.recon import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# initialize datasets and dataloaders
# dataset_num can be equal to 3,4,5,6,7, or 8 only

dataset_num = 4
dataset_dir = '../datasets/' + str(dataset_num if dataset_num != 8 else 1)
dataset_split_dir = '../datasets_splits/' + str(dataset_num)
sample_rate = 2000
dataset = AudioDataset(dataset_dir,
                       dataset_split_dir,
                       'test',
                       sample_rate)

# initialize reconstruction network and load parameters

net = Autoencoder().to(device)
num_epochs = 20
pt_filename = 'dataset'+str(dataset_num)+'_'+str(num_epochs)+'epochs.pt'
param_path = 'parameters/recon/' + pt_filename
net.load_state_dict(torch.load(param_path))

# sample an audio signal and reconstruct it

idx = torch.randint(0,len(dataset),(1,)).item()
x,label = dataset[idx]