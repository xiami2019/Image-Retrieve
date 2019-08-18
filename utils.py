import argparse
import itertools
import torch
import math
import os
import re
import sys
import getpass
import random
import pickle
import subprocess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from optim import get_optimizer, AdamInverseSqrtWithWarmup
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import Sampler
from logger import create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' % getpass.getuser()

def initialize_exp(params):
    '''
    Initialize the experience
    - dump parameters
    - create a logger
    '''
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    #get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    #check experiment name
    assert len(params.exp_name.strip()) > 0

    #create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    params.save_path = os.path.join(params.dump_path, 'checkpoint.pth')
    params.save_best = os.path.join(params.dump_path, 'Best_model.pth')
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger

def get_dump_path(params):
    '''
    Create a directory to store the experiment.
    '''
    dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    #create the sweep path
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()
    
    #create a job ID
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()

def bool_flag(s):
    '''
    Parse boolean arguments from the command line.
    '''
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def combination(iterable, r):
    pool = list(iterable)
    n = len(pool)
    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)


def get_triplets(labels):
    labels = labels.cpu().data.numpy()
    triplets = []
    for label in set(labels):
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combination(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))

def triplet_hashing_loss(image_embedding, image_labels, margin=1):
    
    triplets = get_triplets(image_labels)
    # triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    # loss = triplet_loss(image_embedding[triplets[:, 0]], image_embedding[triplets[:, 1]], image_embedding[triplets[:, 2]])

    # return loss

    ap_distances = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 2]]).pow(2).sum(1)

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()

def cal_result(data_loarder, model, params):
    binary_code = []
    labels = []
    with torch.no_grad():
        for image, label in data_loarder:
            labels.append(label)
            output = model(image.cuda())
            binary_code.append(output.data.cpu())
    return torch.sign(torch.cat(binary_code)), torch.cat(labels)

def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

def save_model(model, save_path):
    model.save(save_path)

def load_model(model, save_path):
    model.load(save_path)

def optimize(optimizer, parameters, params, loss):
    optimizer.zero_grad()
    loss.backward()
    if params.clip_grad_norm > 0:
        clip_grad_norm_(parameters, params.clip_grad_norm)
    optimizer.step()

class categoryRandomSampler(Sampler):
    def __init__(self, numBatchCategory, targets, batch_size):
        """
        This sampler will sample numBatchCategory categories in each batch.
        """
        self.batch_size = batch_size
        self.num_samples = len(targets)
        self.numBatchCategory = numBatchCategory
        self.num_categories = max(targets)
        self.category_idxs = {}
        self.categorys = list(range(1, self.num_categories+1))

        for i in range(1, self.num_categories+1):
            self.category_idxs[i] = []

        for i in range(self.num_samples):
            self.category_idxs[int(targets[i])].append(i)

    def __iter__(self):
        num_batches = self.num_samples // self.batch_size
        selected = []

        for i in range(num_batches):
            batch = []
            random.shuffle(self.categorys)
            categories_selcted = self.categorys[:self.numBatchCategory]
            # categories_selcted = np.random.randint(self.num_categories, size=self.numBatchCategory)

            for j in categories_selcted:
                random.shuffle(self.category_idxs[j])
                batch.extend(self.category_idxs[j][:int(self.batch_size // self.numBatchCategory)])

            random.shuffle(batch)

            selected.extend(batch)

        # print('--------------------------------------------', countn / (countp + countn) * 1.0)

        return iter(torch.LongTensor(selected))

    def __len__(self):
        return self.num_samples
