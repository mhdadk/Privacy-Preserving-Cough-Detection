import torch
import torchaudio
import optuna
import pickle

from models.recon3 import Autoencoder
from torch_datasets.AudioDataset import AudioDataset

def run_batch(x,spec,net,mode,loss_func,alpha,optimizer,device):
    
    # move to GPU if available
    
    x = x.to(device)
    
    x = spec(x)
    
    x = torchaudio.functional.magphase(x)[0]
    
    # scale each example in the batch to interval [0,1]
    
    scale_factor = x.amax(dim=(2,3))[(..., ) + (None, ) * 2]
    
    x_scaled = x / scale_factor
    
    with torch.set_grad_enabled(mode == 'train'):
        
        # compute reconstruction of input signal
        
        x_hat = net(x_scaled)
        
        # re-scale back to normal values
        
        x_hat = x_hat * scale_factor
        
        # compute reconstruction loss
        
        recon_loss = loss_func(x_hat,x,alpha)
        
        if mode == 'train':
        
            # compute gradients of reconstruction loss with respect to
            # parameters
            
            recon_loss.backward()
            
            # update parameters using these gradients
            
            optimizer.step()
            
            # zero the accumulated parameter gradients
            
            optimizer.zero_grad()
    
    return recon_loss.item()

def run_epoch(mode,net,spec,dataloader,optimizer,loss_func,loss_func_alpha,
              device):
    
    if mode == 'train':
        # print('Training...')
        net.train()
    else:
        # print('\nValidating...')
        net.eval()
    
    # to compute average reconstruction loss per sample
    
    total_recon_loss = 0
    
    for i,x in enumerate(dataloader):
        
        # track progress
        
        # print('\rProgress: {:.2f}%'.format((i+1)/len(dataloader)*100),
        #       end='',flush=True)
        
        # train or validate over the batch
        
        recon_loss = run_batch(x,spec,net,mode,loss_func,loss_func_alpha,
                               optimizer,device)
        
        # record running statistics
        
        total_recon_loss += recon_loss
    
    recon_loss_per_sample = total_recon_loss / len(dataloader.dataset)
    
    return recon_loss_per_sample

def loss_func(x_hat,x,alpha = 1):
    
    # spectral convergence
    
    num = torch.linalg.norm(x - x_hat, ord = 'fro', dim = (2,3)).squeeze()
    den = torch.linalg.norm(x, ord = 'fro', dim = (2,3)).squeeze()
    spec_conv = torch.div(num,den)
    
    # log-scale STFT magnitude loss
    
    eps = 1e-10
    log_loss = torch.linalg.norm(torch.log(x + eps) - torch.log(x_hat + eps),
                                 ord = 1, dim = (2,3)).squeeze()
    
    return torch.sum(spec_conv + alpha * log_loss)

def objective(trial):
    
    # initialize network
    
    inst_norm = False #trial.suggest_categorical('inst_norm',[True,False])
    num_channels = trial.suggest_categorical('num_channels',
                                             [8,16,32,64])
    net = Autoencoder(inst_norm = inst_norm,
                      num_channels = num_channels).to(device)
    
    # initialize optimizer
    
    optimizer_name = trial.suggest_categorical('optimizer',
                                               ['Adam','RMSprop'])
    optimizer_func = getattr(torch.optim,optimizer_name)
    
    lr = trial.suggest_float('lr',1e-5,1e-2,log=True)
    momentum = trial.suggest_float('momentum',0.5,0.999)
    
    if optimizer_name == 'Adam':
        beta1 = trial.suggest_float('adam_beta1',0.4,0.9)
        beta2 = trial.suggest_float('adam_beta2',0.8,0.999)
        optimizer = optimizer_func(net.parameters(),
                                   lr = lr,
                                   betas=(beta1,beta2),
                                   eps = 1e-08,
                                   weight_decay = 0,
                                   amsgrad = False)
    elif optimizer_name == 'RMSprop':
        rmsprop_alpha = trial.suggest_float('rmsprop_alpha',0.8,0.99)
        optimizer = optimizer_func(net.parameters(),
                                   lr = lr,
                                   alpha = rmsprop_alpha,
                                   eps = 1e-08,
                                   weight_decay = 0,
                                   momentum = momentum,
                                   centered = False)
    else:
        optimizer = optimizer_func(net.parameters(),
                                   lr = lr,
                                   momentum = momentum,
                                   dampening = 0,
                                   weight_decay = 0,
                                   nesterov = False)
    
    # initialize dataloaders
    
    batch_size = trial.suggest_categorical('batch_size',
                                            [8,16,32,64])
    
    for mode in ['train','val']:
        
        dataset = AudioDataset(raw_data_dir,window_length,sample_rate,mode,
                               normalize = True, only_speech = True)
        
        dataloaders[mode] = torch.utils.data.DataLoader(
                                   dataset = dataset,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   **dl_config)
    
    loss_func_alpha = trial.suggest_discrete_uniform('loss_func_alpha',
                                                     1,10,1)
    
    for epoch in range(num_epochs):
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        # train for an epoch
    
        train_loss = run_epoch('train',
                               net,
                               spec,
                               dataloaders['train'],
                               optimizer,
                               loss_func,
                               loss_func_alpha,
                               device)
        
        print('\nAverage Training loss: {:.4f}'.format(train_loss))
        
        # validate for an epoch
        
        val_loss = run_epoch('val',
                             net,
                             spec,
                             dataloaders['val'],
                             optimizer,
                             loss_func,
                             loss_func_alpha,
                             device)
        
        print('\nAverage Validation Loss: {:.4f}'.format(val_loss))
        
        trial.report(val_loss,epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

# for reproducibility

torch.manual_seed(42)

# check if GPU is available

use_cuda = torch.cuda.is_available()

# to put tensors on GPU if available

device = torch.device('cuda' if use_cuda else 'cpu')

# number of epochs for each trial

num_epochs = 5

# for dataloaders

raw_data_dir = 'data/raw'
window_length = 1.5 # seconds
sample_rate = 16000
dataloaders = {}
dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# intialize spectrogram

win_length_sec = 0.012
win_length = int(sample_rate * win_length_sec)
# need 50% overlap to satisfy constant-overlap-add constraint to allow
# for perfect reconstruction using inverse STFT
hop_length = int(sample_rate * win_length_sec / 2)
spec = torchaudio.transforms.Spectrogram(n_fft = 512,
                                         win_length = win_length,
                                         hop_length = hop_length,
                                         pad = 0,
                                         window_fn = torch.hann_window,
                                         power = None,
                                         normalized = False,
                                         wkwargs = None).to(device)

# tune hp using optuna

pruner = optuna.pruners.HyperbandPruner(max_resource = num_epochs)
sampler = optuna.samplers.TPESampler(seed = 42,
                                     multivariate = True)
study = optuna.create_study(study_name = 'recon_hp_tuning',
                            direction = 'minimize',
                            pruner = pruner,
                            sampler = sampler)
if __name__ == '__main__':
    
    study.optimize(objective,
                   n_trials = 50,
                   n_jobs = 1,
                   gc_after_trial = True)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    pickle.dump(study,open('results/study.pkl','wb'))
    
