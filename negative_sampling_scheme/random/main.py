# -*- coding: utf-8 -*-

"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import os
import numpy as np 
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from sklearn.model_selection import KFold
from utils import Construct_G,Construct_H,one_hot_tensor,Global_Normalize,loss_function,get_metric,get_negative_samples
from data_reading import read_data
from models import xavier_init, GraphConvolution, GCN_decoder
import torch 
import torch.nn as nn
import torch.nn.functional as F
from train_test_split import kf_split
from sklearn.metrics import roc_curve, auc,average_precision_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import argparse
torch.cuda.manual_seed(477)

def parse_args():
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-hgcn_dim', '--hgcn_dim_topofallfeature', type=int, nargs='?', default=2800,help='defining the size of hidden layer of GCN.')
    parser.add_argument('-dropout', '--dropout_topofallfeature', type=float, nargs='?', default=0.5,help='ratio of drop the graph nodes.')
    parser.add_argument('-epoch_num', '--epoch_topofallfeature', type=int, nargs='?', default=1200,help='number of epoch.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    device = 'cuda:1'
    hgcn_dim = args.hgcn_dim_topofallfeature
    dropout = args.dropout_topofallfeature
    epoch_num = args.epoch_topofallfeature

    print('hgcn_dim:',hgcn_dim)
    print('dropout:',dropout)    
    print('epch_num:',epoch_num)
    
    lr = 0.00005     
    topk = 1      #control the number of negative sample 

    data_folder = "/home/jiani.ma/dbMVGCNAE/dataset/DTI/"
    drug_sim_path = "drug_affinity_matrix.xlsx"
    target_sim_path = "protein_affinity_matrix.xlsx"   
    DTI_path = "dti_mat.xlsx"
    
    """
    data reading
    """
    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)    
    """
    globalize the drug affinity matrix and target affinity matrix 
    """    
    SR = Global_Normalize(SR)
    SD = Global_Normalize(SD)    
    
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    A_orig_list = A_orig.flatten()     
    
    #drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)
        
    """
    performing k-fold
    """
    n_splits = 10
    train_all, test_all = kf_split(known_sample,n_splits)    
    
    """
    all those unknown samples are negative samples:
        find their indice in A_orig 
        negative_index_arr : numpy format
        negative_index : tensor format 
    """
    negtive_index_arr = np.where(A_orig_arr==0)[0]
    negative_index = torch.LongTensor(negtive_index_arr)    

    overall_auroc = 0 
    overall_aupr = 0
    overall_f1 = 0 
    overall_acc = 0
    overall_recall = 0    
    overall_specificity = 0
    overall_precision = 0 
            
    for fold_int in range(1):
        print('fold_int:',fold_int)        
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]    
        
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]        
        A_train_tensor = torch.LongTensor(A_train)
        A_test_tensor = torch.LongTensor(A_test)
        
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1        
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1                                
        A_train_mask = A_train_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_test_mask = A_test_list.reshape((A_orig.shape[0],A_orig.shape[1]))        
        A_unknown_mask = 1 - A_orig            
        A_train_mat = A_train_mask    
                
        # G is the normalized adjacent matrix 
        # H is the feature embedding matrix for updation                
        G = Construct_G(A_train_mat,SR,SD).to(device)
        H = Construct_H(A_train_mat,SR,SD).to(device)        
        # sample the negative samples         
        """
        select negative samples 
        """        
        train_neg_mask = get_negative_samples(A_train_mask,A_orig_arr,topk)
        train_negative_index = np.where(train_neg_mask.flatten() ==1)[0]
        training_negative_index = torch.tensor(train_negative_index)

        train_W = torch.randn(hgcn_dim, hgcn_dim).to(device)  
        train_W = nn.init.xavier_normal_(train_W)        
        # initizalize the model 
        gcn_model = GCN_decoder(in_dim=H.size(0),hgcn_dim=hgcn_dim,train_W = train_W,dropout=dropout).to(device)    
        # choose the optimizer
        gcn_optimizer = torch.optim.Adam(list(gcn_model.parameters()),lr=lr)                
        gcn_model.train()       
        for epoch in range(epoch_num):                             
            #prediction results            
            H_up,A_hat = gcn_model(H,G,drug_num,target_num)                      
            A_hat_list = A_hat.view(1,-1)            
            train_sample = A_hat_list[0][A_train_tensor]                        
            train_score = torch.sigmoid(train_sample)                    
            nega_sample = A_hat_list[0][training_negative_index]
            nega_score = torch.sigmoid(nega_sample)
            # calculate the loss                         
            loss = loss_function(train_score,nega_score,drug_num,target_num)            
            los_ = loss.detach().item()           
            if epoch % 10 ==0:
                print('loss:',los_)                            
            gcn_optimizer.zero_grad()
            loss.backward()
            gcn_optimizer.step()          
        gcn_model.eval()          
        test_neg_mask = get_negative_samples(A_test_mask,A_orig_arr,topk)        
        HR = H_up[0:drug_num]
        HD = H_up[drug_num:(drug_num+target_num)] 
        for i in range(HR.shape[0]):
            for j in range(HR.shape[1]):
                print('HR:',HR[i][j].item())
            
        for m in range(HD.shape[0]):
            for n in range(HD.shape[1]):
                print('HD:',HD[m][n].item())    
            
        """
        posive samples  
        """
        pos_drug_target_indice = np.where(A_test_mask==1)
        pos_drug_indice = pos_drug_target_indice[0]
        pos_target_indice = pos_drug_target_indice[1]
        for c in range(len(pos_drug_indice)): 
            print('pos_drug_index:',pos_drug_indice[c])
        for d in range(len(pos_target_indice)):
            print('pos_target_index:',pos_target_indice[d])        
        """
        negative samples 
        """
        neg_drug_target_indice = np.where(test_neg_mask==1)
        neg_drug_indice = neg_drug_target_indice[0]
        neg_target_indice = neg_drug_target_indice[1]
        for h in range(len(neg_drug_indice)):
            print('neg_drug_index:',neg_drug_indice[h])
        for z in range(len(neg_target_indice)):
            print('neg_target_index:',neg_target_indice[z])    






    
      


     
        
    
    
    
    
    
    
    
    
    
    
    
