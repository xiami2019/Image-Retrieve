#Author by Cqy2019
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model import RetrievalModel
from datasets import CUB_200_2011, Stanford_Dog
from utils import *
from optim import get_optimizer
from tensorboardX import SummaryWriter

def get_parser():
    '''
    Generate a parameters parse
    '''
    parser = argparse.ArgumentParser(description='CUB image retrieve')
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',
                        help='dataset name (CUB_200_2011 or Stanford_Dogs)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch_size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers for data loading')
    parser.add_argument('--root', type=str, default='/home/disk1/chengqinyuan/datasets',
                        help='location of data')
    parser.add_argument('--epochs', type=int, default=4000,
                        help='epochs for training')
    parser.add_argument('--eval_only', type=bool_flag, default=False,
                        help='decide to train or evaluate')
    parser.add_argument('--optimizer', type=str, default='adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001',
                        help='choose which optimizer to use')
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--code_size", type=int, default=32,
                        help="size of binary code")
    parser.add_argument("--load_model", type=str, default='',
                        help="location of saved model")
    parser.add_argument("--triplet_margin", type=int, default=8,
                        help="margin when calculate triplet loss")

    
    return parser

def main(params):

    logger = initialize_exp(params)
    writer = SummaryWriter(os.path.join(params.dump_path, 'runs'))

    if params.dataset_name == 'CUB_200_2011':
        params.root = os.path.join(params.root, 'CUB_200_2011')
        dataset = CUB_200_2011
    elif params.dataset_name == 'Stanford_Dogs':
        params.root = os.path.join(params.root, 'Stanford_Dogs')
        dataset = Stanford_Dog
    else:
        logger.info('Dataset %s does not exsist.' % params.dataset_name)

    logger.info('loading %s dataset' % params.dataset_name)
    dataloaders = {
        'train': DataLoader(dataset(params.root, if_train=True), batch_size=params.batch_size,
                            shuffle=True, num_workers=params.num_workers),
        'test': DataLoader(dataset(params.root, if_train=False), batch_size=params.batch_size,
                           shuffle=False, num_workers=params.num_workers),
        'database': DataLoader(dataset(params.root, if_train=True), batch_size=params.batch_size,
                            shuffle=False, num_workers=params.num_workers)
    }

    #create neural network
    training_network = RetrievalModel(params=params)
    training_network.cuda()

    #create optimizer
    optimizer = get_optimizer(training_network.parameters(), params.optimizer)
    # optimizer = torch.optim.Adam(training_network.parameters())
    # optimizer = torch.optim.SGD(training_network.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if params.eval_only is False:
        logger.info('Start Training')
        max_map = 0
        for epoch in range(params.epochs):
            training_network.train()
            total_batches = len(dataloaders['train'])
            logger.info('============ Starting epoch %i ... ============' % epoch)
            count = 0
            total_loss = 0
            for index, (images, labels) in enumerate(dataloaders['train']):
                optimizer.zero_grad()
                images = images.cuda()
                labels = labels.cuda()
                images.requires_grad = True
                image_embeddings = training_network(images)
                triplet_loss = triplet_hashing_loss(image_embeddings, labels, margin=params.triplet_margin)
                logger.info('Batch %i/%i: loss: %f' % (index + 1, total_batches, triplet_loss.item()))
                total_loss += triplet_loss.item()
                count +=1 
                writer.add_scalar('train', triplet_loss.item(), index + epoch * params.batch_size)
                triplet_loss.backward()
                optimizer.step()
                scheduler.step()
            logger.info('============ End of epoch %i ============' % epoch)
            
            if epoch % 100 == 0:
                #caculate map
                training_network.eval()
                train_binary, train_label = cal_result(dataloaders['database'], training_network, params)
                test_binary, test_label = cal_result(dataloaders['test'], training_network, params)
                mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
                mAP = float(mAP)
                logger.info('mAP: %f' % mAP)
                if mAP > max_map:
                    max_map = mAP
                    save_model(training_network, params.save_best)
                save_model(training_network, params.save_path)
            logger.info('\nEpoch %i: avg loss: %f\n' % (epoch, total_loss / count))
    
    if params.eval_only:
        logger.info('Start Testing')
        assert os.path.isfile(params.load_model)
        load_model(training_network, params.load_model)
        training_network.eval()
        train_binary, train_label = cal_result(dataloaders['database'], training_network, params)
        test_binary, test_label = cal_result(dataloaders['test'], training_network, params)
        mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
        logger.info("mAP: %f" % mAP)
    
    writer.close()

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)