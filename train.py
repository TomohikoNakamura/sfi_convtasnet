import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from model.tasnet import MultiVariableFIRConvTasNet
from utility.logger import Logger
from utility.loss import sdr_objective
from utility.ranger import Ranger
from utility.sgdr_learning_rate import SGDRLearningRate

def define_model(device, args):
    if args.model_type == "sfi_convtasnet":
        network = MultiVariableFIRConvTasNet(args).to(device)
    else:
        raise NotImplementedError(f"Unknown model type [{args.model_type}]")
    # print(network)
    return network

def train_step(network, batch, device):
    batch = tuple([s.to(device) for s in b] for b in batch)
    loss, stats = network(*batch)
    loss = loss.mean()
    stats = stats.mean(0).cpu().detach().numpy()
    return loss, stats

def eval_step(network, batch, device):
    batch = tuple([s.to(device) for s in b] for b in batch)
    mix, separated = batch
    outputs = network.inference(mix, n_chunks=network.args.n_chunks)
    objectives = [sdr_objective(o, s) for o, s in zip(outputs, separated)]
    objectives = torch.cat(objectives, 0).cpu().numpy()
    return objectives

@hydra.main(config_path="configs/config.yaml")
def app(args: DictConfig) -> None:
    ## model
    args.seed = int(args.seed)
    args.directory = os.getcwd()
    print(args.pretty())

    # Move
    os.chdir(hydra.utils.get_original_cwd())

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create the model
    network = define_model(device, args)

    # Load data
    if args.debug: from dataset_stub import MusicDataset
    else: from dataset import MusicDataset

    train_data = MusicDataset(args.train_data, args.sampling_rate / 1000, args.stages_num, sample_length=args.time_length, shuffle_p=args.shuffle_p, is_train=True, verbose=True, num_workers=args.threads)
    eval_data = MusicDataset(args.validation_data, args.sampling_rate / 1000, args.stages_num, sample_length=args.time_length, is_train=False, verbose=True)
    print("############################")
    print(f"    # of train data: {len(train_data)}")
    print(f"    # of valid data: {len(eval_data)}")
    print("############################")

    print("Data loaded successfully")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=train_data.get_collate_fn())
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=args.threads)
    
    optimizer = Ranger(filter(lambda p: p.requires_grad, network.parameters()), weight_decay=args.weight_decay)
    decay = SGDRLearningRate(optimizer, args.learning_rate, t_0=args.sgdr_period, mul=0.85)
    logger = Logger()

    # Optionally load from a checkpoint
    if args.checkpoint is not None:
        resume_path = Path(f"{args.directory}/{args.checkpoint}")
        if resume_path.exists():
            print(f"Resume from {args.directory}/{args.checkpoint}")
            state = torch.load(f"{args.directory}/{args.checkpoint}")
            optimizer.load_state_dict(state['optimizer'])
            network.load_state_dict(state['state_dict'])
            initial_epoch = state['epoch'] + 1
            steps = state['steps']
        else:
            print(f'Checkpoint file is not found [{resume_path}].')
            initial_epoch, steps = 0, 0
    else:
        initial_epoch, steps = 0, 0

    # Optionally distribute the model across more GPUs
    raw_network = network
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network)
        network = network.to(device)

    # Start training
    best_validation_objective = float('-inf')

    for epoch in range(initial_epoch, args.epochs):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:

            #
            # TRAIN EPOCH
            #

            network.train()

            running_objectives, batches_done = np.zeros(args.stages_num*4 + 3), 0
            start_time = time.time()

            for i_batch, batch in enumerate(train_loader):

                steps += 1
                if decay(steps):
                    break  # evaluate after decay cycle resets

                loss, stats = train_step(network, batch, device)

                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), args.clip_gradient)
                optimizer.step()
                optimizer.zero_grad()
                del loss

                with torch.no_grad():
                    batches_done += 1
                    running_objectives += stats

                    progress = 100 * i_batch//len(train_loader)
                    average_objectives = running_objectives / batches_done
                    if (i_batch+1) % 100 == 0:
                        logger.log_train_progress(epoch, average_objectives, args.stages_num, decay.learning_rate, progress)

            average_objectives = running_objectives / batches_done
            logger.log_train(epoch, average_objectives, args.stages_num, int(time.time() - start_time), log_file)


            #
            # EVALUATE EPOCH
            #

            network.eval()
            running_stats, batches_done = np.zeros(args.stages_num*4), 0
            
            with torch.no_grad():
                # start_time = time.time()
                for i_batch, batch in enumerate(eval_loader):
                    stats = eval_step(raw_network, batch, device)
                    running_stats += stats
                    batches_done += 1
                average_stats = running_stats / batches_done
                logger.log_dev(average_stats, args.stages_num, decay.learning_rate, log_file)

                state = {
                    'epoch': epoch,
                    'state_dict': raw_network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'args': args
                }
                objective = average_stats[-5:-1].mean()

                if objective > best_validation_objective:
                    best_validation_objective = objective
                    torch.save(state, f'{args.directory}/best_checkpoint')
                torch.save(state, f'{args.directory}/last_checkpoint')

if __name__ == "__main__":
    app()

