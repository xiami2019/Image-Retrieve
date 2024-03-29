#Author by Cqy2019
import argparse
from ast import parse
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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tensorboardX import SummaryWriter

def get_parser():
    '''
    Generate a parameters parse
    '''
    parser = argparse.ArgumentParser(description='CUB image retrieve')
    parser.add_argument('--dump_path', type=str, default='./dumped/',
                        help='Experiment dump path')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment name')
    parser.add_argument('--exp_id', type=str, default='',
                        help='Experiment ID')
    parser.add_argument('--dataset_name', type=str, default='CUB_200_2011',
                        help='dataset name (CUB_200_2011 or Stanford_Dogs)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch_size for training')
    parser.add_argument('--root', type=str, default='datasets',
                        help='location of data')
    parser.add_argument('--epochs', type=int, default=4000,
                        help='epochs for training')
    parser.add_argument('--eval_only', type=bool_flag, default=False,
                        help='decide to train or evaluate')
    parser.add_argument('--optimizer', type=str, default='adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001',
                        help='choose which optimizer to use')
    parser.add_argument('--code_size', type=int, default=32,
                        help='size of binary code')
    parser.add_argument('--load_model', type=str, default='',
                        help='location of saved model')
    parser.add_argument('--triplet_margin', type=int, default=8,
                        help='margin when calculate triplet loss')
    parser.add_argument('--category_sampler', type=bool_flag, default=False,
                        help='Whether to use category sampler when combine triplet')
    parser.add_argument('--category_number', type=int, default=0,
                        help='number of category when use category sampler')
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--model', type=str, default='vit', choices=['resnet18', 'vit'])
    
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

    logger.info('Loading %s dataset' % params.dataset_name)
    if params.category_sampler:
        assert params.category_number > 0
        all_targets = []
        temp_dataloader = DataLoader(dataset(params.root, if_train=True), batch_size=params.batch_size,
                                shuffle=True)
        for _, labels in temp_dataloader:
            all_targets.extend(labels)

        logger.info('Generating sampler')
        category_sampler = categoryRandomSampler(params.category_number, all_targets, params.batch_size)
        dataloaders = {
            'train': DataLoader(dataset(params.root, if_train=True), batch_size=params.batch_size,
                                shuffle=False, sampler=category_sampler),
            'test': DataLoader(dataset(params.root, if_train=False), batch_size=params.batch_size,
                            shuffle=False),
            'database': DataLoader(dataset(params.root, if_train=True, if_database=True), batch_size=params.batch_size,
                                shuffle=False)
        }
    else:
        dataloaders = {
            'train': DataLoader(dataset(params.root, if_train=True), batch_size=params.batch_size,
                                shuffle=True),
            'test': DataLoader(dataset(params.root, if_train=False), batch_size=params.batch_size,
                            shuffle=False),
            'database': DataLoader(dataset(params.root, if_train=True, if_database=True), batch_size=params.batch_size,
                                shuffle=False)
        }

    #create neural network
    training_network = RetrievalModel(params=params)
    training_network.cuda()

    #create optimizer
    # optimizer = get_optimizer(training_network.parameters(), params.optimizer)
    # optimizer = torch.optim.Adam(training_network.parameters())
    # optimizer = torch.optim.SGD(training_network.parameters(), lr=0.001, momentum=0.9)
    num_train_steps = params.epochs * len(dataloaders['train'])
    num_warmup_steps = params.warmup_ratio * num_train_steps
    optimizer = AdamW(training_network.parameters(), lr=params.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

    if params.eval_only is False:
        logger.info('Start Training')
        max_map = 0
        early_stopping_count = 0

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
            logger.info('\nEpoch %i: avg loss: %f\n' % (epoch, total_loss / count))
            
            if epoch % params.eval_epoch == 0 or epoch == params.epochs - 1:
                #caculate map
                training_network.eval()
                train_binary, train_label = cal_result(dataloaders['database'], training_network, params)
                test_binary, test_label = cal_result(dataloaders['test'], training_network, params)
                
                mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
                mAP = float(mAP)
                logger.info('mAP: %f' % mAP)
                if mAP >= max_map:
                    early_stopping_count = 0
                    max_map = mAP
                    save_model(training_network, params.save_best)
                else:
                    early_stopping_count += 1
                    if early_stopping_count > params.early_stopping:
                        break
                
                save_model(training_network, params.save_path)
        logger.info('============ Trainig is over ============')
    
    if params.eval_only:
        logger.info('Start Testing')
        assert os.path.isfile(params.load_model)
        
        load_model(training_network, params.load_model)
        training_network.eval()
        train_binary, train_label = cal_result(dataloaders['database'], training_network, params)
        test_binary, test_label = cal_result(dataloaders['test'], training_network, params)
        
        mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
        logger.info('mAP: %f' % mAP)
    
    writer.close()

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)