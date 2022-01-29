# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:28:56 2021

@author: Hui Hu
"""
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
import random
from sklearn.model_selection import train_test_split
import pickle as pkl
from scipy.spatial import distance_matrix

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id[:200]:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map


def load_pokec(dataset,sens_attr,predict_attr,path):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))
    
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #67796*279
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove("completion_percentage")
    header.remove("AGE")
    header.remove(predict_attr) # remove predictable feature
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) #non-sensitive
    labels = idx_features_labels[predict_attr].values #array([-1,  0,  1,  2,  3, 4], dtype=int64)
    label_idx = np.where(labels<0)[0]
    labels[label_idx] = np.max(labels)+1 #convert negative label to positive
    sens = idx_features_labels[sens_attr].values
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),    
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)#-1,0,1,2,3,4
    sens = torch.LongTensor(sens)
    return adj, features, labels, sens

def load_credit(dataset,sens_attr,predict_attr,path):
    print('Loading {} dataset from {}'.format(dataset,path))
    
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)  
   # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) #non-sensitive
    labels = idx_features_labels[predict_attr].values  
    sens = idx_features_labels[sens_attr].values
    
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    return adj, features, labels, sens

def load_bail(dataset,sens_attr,predict_attr,path):
    print('Loading {} dataset from {}'.format(dataset,path))
    
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)  
   # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) #non-sensitive
    labels = idx_features_labels[predict_attr].values  
    sens = idx_features_labels[sens_attr].values
    
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    return adj, features, labels, sens

def load_german(dataset,sens_attr,predict_attr,path):
    print('Loading {} dataset from {}'.format(dataset,path))
    
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset))) #67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan') 
   # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)
     # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
    sens = idx_features_labels[sens_attr].values.astype(np.int64)
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32) #non-sensitive
    labels = idx_features_labels[predict_attr].values  
    label_idx = np.where(labels==-1)[0]
    labels[label_idx] = 0 #convert negative label to positive
   
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    return adj, features, labels, sens
    

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return (features - min_values).div(max_values-min_values) 

def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_train_val_test(features, val_size=0.2, test_size=0.3,seed=None):
   
    if seed is None:
        np.random.seed(seed)

    idx = np.arange(features.shape[0])
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,                                                  
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size)                                         
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          #train_size=train_size/ (train_size + val_size),
                                          test_size=(val_size / (train_size + val_size)))                                   
    return idx_train, idx_val, idx_test


def split_train(idx_train,sensitive_size=0.5, nosensitive_size=0.5,seed=None):
    
    idx_sensitive, idx_nonsensitive = train_test_split(idx_train,
                                          random_state=None,
                                          train_size=sensitive_size,
                                          test_size=nosensitive_size)   
    return idx_sensitive, idx_nonsensitive

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def data_preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device=None):
    if preprocess_adj:
        adj = normalize(adj) 
    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())
    if preprocess_feature:
        features = feature_norm(features)
    return adj.to(device), features.to(device), labels.to(device)