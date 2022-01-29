# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 09:39:45 2021

@author: Hui Hu
"""
import argparse
import numpy as np
import torch
from load_data import load_pokec,load_credit,load_german,load_bail,split_train,get_train_val_test
from load_data import data_preprocess
from Original_GCN import GCN
from model import DPGCN
from scipy.sparse import csr_matrix

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed1', type=int, default=100, help='Random seed.')
parser.add_argument('--seed2', type=int, default=50, help='Random seed.')
parser.add_argument('--epoch1', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--epoch2', type=int, default=200,
                    help='Number of epochs to infer.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--lr_w1', type=float, default=0.001,
                    help='Initial learning rate for S1.')
parser.add_argument('--lr_w2', type=float, default=0.001,
                    help='Initial learning rate for S2.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='pokec_z',
                    choices=['pokec_z','pokec_n','credit','german','bail'])
parser.add_argument('--S1', type=int, default=1, help='steps for S1 optimization')
parser.add_argument('--S2', type=int, default=1, help='steps for S2 optimization')
parser.add_argument('--inner_steps', type=int, default=1, help='steps for inner optimization')
parser.add_argument('--alpha', type=float, default=1, help='weight of loss_re')
parser.add_argument('--beta', type=float, default=5, help='weight of loss_orth')
parser.add_argument('--gamma', type=float, default=1, help='weight of loss_gcn')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)
if args.cuda:
    torch.cuda.manual_seed(args.seed1)

# Load data
if args.dataset == 'pokec_z':
    dataset = 'region_job'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    path="../dataset/pokec/"
    adj, features, labels, sens  = load_pokec(dataset, sens_attr,predict_attr,path=path)  #label: 0 1 2 3 4 5
    sen_attr_index =2

elif args.dataset == 'pokec_n':
    dataset = 'region_job_2'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field"
    path="../dataset/pokec/"
    adj, features, labels, sens  = load_pokec(dataset, sens_attr,predict_attr,path=path) #label: 0 1
    sen_attr_index =2
    
elif args.dataset == 'credit':
    dataset = 'credit'
    sens_attr = 'Age'
    predict_attr = 'NoDefaultNextMonth'
    path="../dataset/credit/"
    adj, features, labels, sens  = load_credit(dataset, sens_attr,predict_attr,path=path) #label: 0 1
    sen_attr_index =2

elif args.dataset == 'german':
    dataset = 'german'
    sens_attr = 'Gender'
    predict_attr = "GoodCustomer"
    path="../dataset/german/"
    adj, features, labels, sens  = load_german(dataset,sens_attr,predict_attr,path=path) #label: 0 1
    sen_attr_index =0
    
elif args.dataset == 'bail':
    dataset = 'bail'
    sens_attr = 'WHITE'
    predict_attr = "RECID"
    path="../dataset/bail/"
    adj, features, labels, sens  = load_bail(dataset,sens_attr,predict_attr,path=path) #label: 0 1
    sen_attr_index =0
    
idx_train, idx_validate, idx_test = get_train_val_test(features, val_size=0.3, test_size=0.2,seed=args.seed1)
idx_sensitive, idx_nosensitive = split_train(idx_train,sensitive_size=0.3, nosensitive_size=0.7,seed=args.seed2)
#model training
torch.manual_seed(args.seed1)
model = GCN(nfeat=features.shape[1], 
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device)
std=1
S1 = torch.normal(0,std, size=(features.shape[1]-1, features.shape[1]),generator=torch.manual_seed(15))
S2 = torch.normal(0,std, size=(features.shape[1]-1, features.shape[1]),generator=torch.manual_seed(10))
adj, features, labels = data_preprocess(adj,csr_matrix(features),labels,preprocess_adj=True,preprocess_feature=True,device=device)
pgcn = DPGCN(model, args, device)
pgcn.fit(features,adj,S1,S2,labels,sens,idx_train, idx_validate,idx_test,idx_sensitive,idx_nosensitive,sen_attr_index)
pgcn.test(features,adj,labels,idx_test,sen_attr_index)  
pgcn.infer(features,adj,sen_attr_index,idx_sensitive,idx_nosensitive,sens)  
