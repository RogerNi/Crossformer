'''
Train a Crossformer model on sequential bytes-string predictions with PyTorch.
This code borrows heavily from https://github.com/HazyResearch/state-spaces.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchsummary import summary

from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

import math

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import wandb

from cross_models.cross_former import Crossformer
from tqdm.auto import tqdm

torch.manual_seed(0)


parser = argparse.ArgumentParser(description='My Crossformer Training')
# Inherited arguments
# parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=1, help='output MTS length (\tau)')
parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

parser.add_argument('--data_dim', type=int, default=8, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

# Optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay')
# Scheduler
parser.add_argument('--scheduler', default="ReduceOnPlateau", choices=['ReduceOnPlateau', 'cosine', 'none'] ,type=str, help='Scheduler to use')
parser.add_argument('--patience', default=3, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=30, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='bin', choices=['mnist', 'cifar10', 'bin', 'npy'], type=str, help='Dataset')
parser.add_argument('--datapath', type=str, help='Path to binary data')
parser.add_argument('--img_size', default=100, type=int, help='Resize image to this size')
parser.add_argument('--sequence_len', default=96, type=int, help='Length of sequence')
parser.add_argument('--color_mode', default='bw', choices=['bw', 'grayscale', 'color'], type=str, help='Color mode to use')
parser.add_argument('--dataset_only', action='store_true', help='Build the dataset only')
# parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
parser.add_argument('--random_split', action='store_true', help='randomly split dataset')
parser.add_argument('--save_data', default='', type=str, help='Path to save binary data')
parser.add_argument('--byte_len', default=1000000, type=int, help='Maximum number of bytes to read from dataset, 0 for unlimited')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64 * 8 * 2, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
# parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
# parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--no_wait_gpu', '-g', action='store_true', help='Do not wait for available GPU')
parser.add_argument('--output_predict', action='store_true', help='Output predictions to fd 3')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.output_predict:
    fd3 = os.fdopen(3, 'wb')

#==============Select ununsed GPU================
import subprocess

def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]
if args.no_wait_gpu:
    free_gpus = [0]
else:
    free_gpus = get_free_gpu_indices()
while not free_gpus:
    exit("No free GPU found")
torch.cuda.set_device(free_gpus[0])
#================================================

# Data
print(f'==> Preparing {args.datapath if args.datapath else args.dataset} data..')

class ByteSequenceDataset(Dataset):
    def __init__(self, data_path, sequence_len, one_hot=False):
        self.data = np.fromfile(data_path, dtype=np.uint8)
        self.data = np.unpackbits(self.data).reshape((-1, 8))
        self.sequence_len = sequence_len
        
    def __len__(self):
        return len(self.data) - self.sequence_len - 1
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.sequence_len]).float(), torch.tensor(self.data[idx+self.sequence_len + 1]).float()

class NpyDataset(Dataset):
    def __init__(self, data_path, sequence_len, img_size):
        self.data = torch.from_numpy(np.load(data_path))
        self.data = self.data[torch.randperm(self.data.size()[0])]
        transform = transforms.Compose([
                      transforms.Resize(img_size),
                      transforms.Lambda(lambda x: (x!=255) if args.color_mode == "bw" else x),
                      transforms.Lambda(lambda x: x.view(1, -1).t()),
                      transforms.Lambda(lambda x: x.int()),
                    ])
        self.ori_data = transform(self.data)
        self.data = self.ori_data.flatten().unsqueeze(1).bitwise_and(2**torch.arange(8).flip(0).unsqueeze(0)).ne(0)
        if args.byte_len > 0:
            self.ori_data = self.ori_data[:args.byte_len]
            self.data = self.data[:args.byte_len]
        self.sequence_len = sequence_len
        
    def __len__(self):
        return len(self.data) - self.sequence_len - 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.sequence_len].float(), self.data[idx+self.sequence_len + 1].float()

    def save_dataset_to_binary(self, path):
        with open(path, 'wb') as f:
            f.write(self.ori_data.numpy().astype('uint8'))
    
class ConcatImageDataset(Dataset):
    def __init__(self, DatasetClass, sequence_len):
        if args.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x>0) if args.color_mode == "bw" else x * 255),
                transforms.Lambda(lambda x: x.view(1, 784).t()),
                transforms.Lambda(lambda x: x.int()),
            ])
        else:
            # cifar10
            if args.color_mode == "bw" or args.color_mode == "grayscale":
                transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
                    transforms.Lambda(lambda x: (x>0) if args.color_mode == "bw" else x * 255),
                    transforms.Lambda(lambda x: x.view(1, 1024).t()),
                    transforms.Lambda(lambda x: x.int()),
                ])
            else:
                # color
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    transforms.Lambda(lambda x: x * 255),
                    transforms.Lambda(lambda x: x.view(1, -1).t()),
                    transforms.Lambda(lambda x: x.int()),
                ])
        datasets = DatasetClass(root='./data', train=True, download=True, transform=transform)
        self.ori_data = torch.cat([datasets[i][0] for i in range(len(datasets))])
        self.data = self.ori_data.flatten().unsqueeze(1).bitwise_and(2**torch.arange(8).flip(0).unsqueeze(0)).ne(0)
        if args.byte_len > 0:
            self.ori_data = self.ori_data[:args.byte_len]
            self.data = self.data[:args.byte_len]
        self.sequence_len = sequence_len
        
    def __len__(self):
        return len(self.data) - self.sequence_len - 1
    
    def __getitem__(self, idx):
        return self.data[idx:idx+self.sequence_len].float(), self.data[idx+self.sequence_len + 1].float()
    
    def save_dataset_to_binary(self, path):
        with open(path, 'wb') as f:
            f.write(self.ori_data.numpy().astype('uint8'))
        
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    if not args.random_split:
        val = Subset(train, range(train_len, len(train)))
        train = Subset(train, range(train_len))
    else:
        train, val = torch.utils.data.random_split(
            train,
            (train_len, len(train) - train_len),
            generator=torch.Generator().manual_seed(42),
        )
    return train, val

#========for P local calculation=================
def pfunc(plocal,r,N):
    q = 1.0-plocal
    
    # Find x10
    x = 0.0
    for j in range(1,11):
        x = 1.0 + (q*(plocal**r)*(x**(r+1.0)))

    # do the equation
    result = (1.0 - plocal*x)
    result = result/((r+1.0 - (r*x))*q)
    try:
        result = result/(x**(N+1))
    except OverflowError:
        # catch OverflowError resulting from large N.
        result = 0.0
    return result

def search_for_p(r,N,iterations=1000, min_plocal=0.0, max_plocal=1.0, tolerance=0.00000001,verbose=False):
    P_local = 0
    # Binary chop search for Plocal
    iteration = 0
    found = False
    
    #vprint(verbose,"SEARCH FOR P")
    #vprint(verbose,f'min {min_plocal}  max {max_plocal} verbose={verbose} r={r} N={N}')
    while (iteration < iterations):
        candidate = (min_plocal + max_plocal)/2.0 # start in the middle of the range
        result = pfunc(candidate,r,N)
        #print ("iteration =",iteration)
        #if verbose:
        #    vprint(verbose,f'candidate {candidate}  min {min_plocal}  max {max_plocal}')
        iteration += 1
        if iteration > iterations:
            found = False
            break
        elif (result > (0.99-tolerance)) and (result < (0.99+tolerance)):
            found = True
            P_local = candidate
            break
        elif result > 0.99:
            min_plocal = candidate
        else:
            max_plocal = candidate

    if (found == False):
        print ("Warning: P_local not found")

    return P_local

#================================================

# Prepare datasets
if args.dataset == 'bin':
    dataset = ByteSequenceDataset(args.datapath, args.sequence_len)
elif args.dataset == 'mnist':
    dataset = ConcatImageDataset(torchvision.datasets.MNIST, args.sequence_len)
elif args.dataset == 'cifar10':
    dataset = ConcatImageDataset(torchvision.datasets.CIFAR10, args.sequence_len)
elif args.dataset == 'npy':
    dataset = NpyDataset(args.datapath, args.sequence_len, args.img_size)
else:
    raise Exception('Unknown dataset')

if args.save_data and args.dataset != 'bin':
    dataset.save_dataset_to_binary(args.save_data)
    
trainset, valset = split_train_val(dataset, val_split=0.1)
d_input = d_output = 8 # correpsonds to 8 bits in 1 byte

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print(f'==> Data loaded. Train: {len(trainset)} Val: {len(valset)}')

# Only prepare (or save) dataset. Exit.
if args.dataset_only:
    print('==> Dataset only. Exiting.')
    exit()

# Model
print('==> Building model..')
model = Crossformer(
    args.data_dim, 
    args.sequence_len, 
    args.out_len,
    args.seg_len,
    args.win_size,
    args.factor,
    args.d_model, 
    args.d_ff,
    args.n_heads, 
    args.e_layers,
    args.dropout, 
    args.baseline
).float()

model = model.to(device)

print(model)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create a lr scheduler
    if args.scheduler == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.2)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 1)

    return optimizer, scheduler

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    bit_correct = 0
    symbol_len = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        first_nonzero_idx = first_nonzero(targets, axis=1, invalid_val=8).min()
        symbol_len = max(symbol_len, 8 - first_nonzero_idx)
        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        predicted = outputs
        total += targets.size(0)
        predicted = (predicted > 0.5).float()
        correct += (predicted == targets).all(dim=1).sum().item()
        local_bit_correct = (predicted == targets)[:,-symbol_len:].sum().item()
        bit_correct += local_bit_correct
        # correct += predicted.eq(targets).sum().item()
        
        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Bit Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*bit_correct/(total*symbol_len), bit_correct, total*symbol_len)
        )
        
    return train_loss/(batch_idx+1), correct/total, bit_correct/(total*symbol_len)

# function to help find symbol length
def first_nonzero(arr, axis, invalid_val=-1):
    mask = (arr!=0).float()
    return torch.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    longest = 0
    local = 0
    total = 0
    symbol_len = 0
    #=========following is for calculating local and global performance as bitstring==========
    bit_correct = 0
    bit_local = 0
    bit_longest = 0
    #=========================================================================================
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            first_nonzero_idx = first_nonzero(targets, axis=1, invalid_val=8).min().item()
            symbol_len = max(symbol_len, 8 - first_nonzero_idx)
            outputs = torch.squeeze(model(inputs))
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            # _, predicted = outputs.max(1)
            predicted = outputs
            total += targets.size(0)
            predicted = (predicted > 0.5).float()
            # write predicted to fd 3 (igonre if not existing)
            if args.output_predict:
                fd3.write(predicted[:,-symbol_len:].cpu().numpy().astype("uint8").tobytes())
            local_correct = (predicted == targets).all(dim=1).sum().item()
            local_bit_correct = (predicted == targets)[:,-symbol_len:].sum().item()
            correct += local_correct
            bit_correct += local_bit_correct
            #=======
            # if local_correct == targets.size(0):
            #     local += local_correct
            # else:
            #     if local > longest:
            #         longest = local
            #     local = 0
            for correction in (predicted == targets).all(dim=1):
                if correction:
                    local += 1
                else:
                    if local > longest:
                        longest = local
                    local = 0
                    
            for correction in (predicted == targets)[:,-symbol_len:].flatten():
                if correction:
                    bit_local += 1
                else:
                    if bit_local > bit_longest:
                        bit_longest = bit_local
                    bit_local = 0
            # correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Longest (bytes): %d | Longest (bits): %d | Bit Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total, longest, bit_longest, 100.*bit_correct/total/symbol_len, bit_correct, total*symbol_len)
            )
    if args.output_predict:
        fd3.flush()
    P_global = correct/total
    if P_global == 0:
        P_prime_global = 1.0 -(0.01**(1.0/total))
    else:
        P_prime_global = min(1.0,P_global + (2.576*math.sqrt((P_global*(1.0-P_global)/(total-1.0))))) 
        
    P_bit_global = bit_correct / total / symbol_len
    if P_bit_global == 0:
        P_prime_bit_global = 1.0 -(0.01**(1.0/total/symbol_len))
    else:
        P_prime_bit_global = min(1.0,P_bit_global + (2.576*math.sqrt((P_bit_global*(1.0-P_bit_global)/(total*symbol_len-1.0))))) 
        
    if local > longest:
        longest = local
    if bit_local > bit_longest:
        bit_longest = bit_local
    
    P_local = search_for_p(longest,total)
    P_bit_local = search_for_p(bit_longest,total*symbol_len)
    
    min_entropy = -math.log(max(P_prime_global,P_local,1.0/(2.0**symbol_len)),2)
    min_entropy_per_bit = min_entropy/symbol_len
    min_entropy_bits = -math.log(max(P_prime_bit_global, P_bit_local, 1/2), 2)
    print(f"Epoch {epoch}\n\tSymbol length: {symbol_len}\n\tP_global {P_global}\n\tP_prime_global {P_prime_global}\n\tLongest bytes: {longest}\n\tP_local {P_local}\n\tmin_entropy_bytes {min_entropy}\n\tmin_entropy_bytes_per_bit {min_entropy_per_bit}\n\tP_bit_global {P_bit_global}\n\tP_prime_bit_global {P_prime_bit_global}\n\tLongest bits: {bit_longest}\n\tP_bit_local {P_bit_local}\n\tmin_entropy_bits {min_entropy_bits}\n\tmin_entropy {min(min_entropy_per_bit, min_entropy_bits)}\n")
    
    return eval_loss/(batch_idx+1), correct/total, bit_correct/(total*symbol_len), min_entropy_per_bit, min_entropy_bits, min(min_entropy_per_bit, min_entropy_bits), P_global, P_bit_global, P_local, P_bit_local

    
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="min-entropy",
    
    # track hyperparameters and run metadata
    config=vars(args),
)

min_min_entropy = 1
pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with record_function("model_train"):
    train_loss, train_acc, train_bit_acc = train()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with record_function("model_eval"):
    val_loss, val_acc, val_bit_acc, min_entropy_from_bytes, min_entropy_from_bits, min_entropy, P_global, P_bit_global, P_local, P_bit_local = eval(epoch, valloader, checkpoint=True)
    # eval(epoch, testloader)
    if args.scheduler != "ReduceOnPlateau":
        scheduler.step()
    else:
        scheduler.step(val_acc)
    print(f"Epoch {epoch} learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
    min_min_entropy = min(min_min_entropy, min_entropy)
    wandb.log({
        "Train Loss": train_loss,
        "Train Accuracy (Bytes)": train_acc,
        "Train Accuracy (Bits)": train_bit_acc,
        "Val Loss": val_loss,
        "Val Accuracy (Bytes)": val_acc,
        "Val Accuracy (Bits)": val_bit_acc,
        "P local": P_local,
        "P bit local": P_bit_local,
        "Min Entropy (Bytes)": min_entropy_from_bytes,
        "Min Entropy (Bits)": min_entropy_from_bits,
        "Min Entropy": min_entropy,
        "Learning Rate": optimizer.state_dict()['param_groups'][0]['lr'],
        "Min of Min Entropy": min_min_entropy,
    })

