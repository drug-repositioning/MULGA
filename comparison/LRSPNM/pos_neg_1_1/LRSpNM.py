#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:10:54 2022
@author: Jiani Ma
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from scipy import linalg
import matplotlib.pyplot as plt
from math import sqrt  
import argparse
from numpy.linalg import norm
from sklearn.model_selection import KFold
from numpy import linalg as LA
from utils import * 
from prepare_data import * 
np.random.seed(1)  

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-lambda_d', '--lambda_d_topofallfeature', type=float, nargs='?', default=20,help='Performing k-fold for cross validation.') 
    parser.add_argument('-lambda_t', '--lambda_t_topofallfeature', type=float, nargs='?', default=20,help='Performing k-fold for cross validation.') 
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, nargs='?', default=10,help='Performing k-fold for cross validation.') 
    parser.add_argument('-mu_1', '--mu_1_topofallfeature', type=float, nargs='?', default=10,help='Performing k-fold for cross validation.') 
    return parser.parse_args() 


if __name__=="__main__":
    m = 1822
    n = 1447    
    data_folder = "/home/jiani.ma/dbMVGCNAE/comparision/LRSPNM/data"
    drug_sim_path = "drug_jaccard_smilarity.xlsx"
    target_sim_path = "protein_cosine_similarity.xlsx"
    DTI_path = "dti_mat.xlsx" 
    
    args = parse_args()
    lambda_d = args.lambda_d_topofallfeature    #(2_2, 2-1,20,21,22)
    lambda_t = args.lambda_t_topofallfeature    #(2_2, 2-1,20,21,22)
    alpha = args.alpha_topofallfeature          #(2_2, 2-1,20,21,22)
    mu_1 = args.mu_1_topofallfeature            #(2_2, 2-1,20,21,22)
    beta = mu_2 = mu_1
    p = 0.8    
    topk = 1
    n_splits = 10
    
    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)     
    Ld = LaplacianMatrix(SR)  # drug 
    Lt = LaplacianMatrix(SD)  # target
    drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)

    kf = KFold(n_splits, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_sample):  
        train_all.append(train_ind) 
        test_all.append(test_ind)
    
    overall_auroc = 0 
    overall_aupr = 0
    overall_f1 = 0 
    overall_acc = 0
    overall_recall = 0    
    overall_specificity = 0
    overall_precision = 0 
    
    A_unknown_mask = 1 - A_orig  
    
    for fold_int in range(10):
        print("fold_int:",fold_int)
        pos_train_id = train_all[fold_int]
        pos_train = known_sample[pos_train_id]
        pos_train_mask_list = np.zeros_like(A_orig_arr)
        pos_train_mask_list[pos_train] = 1 
        pos_train_mask = pos_train_mask_list.reshape((m,n))
        neg_train_mask_candidate = get_negative_samples(pos_train_mask,drug_dissimmat)       
        neg_train_mask = np.multiply(neg_train_mask_candidate, A_unknown_mask)    
        A_train_mat = np.copy(pos_train_mask)
        A_train_mask = np.copy(pos_train_mask) #pos_train_mask + neg_train_mask 
             
        lastW = lastX = lastZ = lastU = lastV = np.zeros((m,n))
        for i in range(100):
            currentW = update_W(lastX,lastU,mu_1,p)
            currentZ = update_Z(alpha,mu_2,A_train_mat,lastV,lastX,A_train_mask)
            currentX = update_X(lambda_d,lambda_t,Ld,Lt,mu_1,mu_2,currentW,currentZ,lastU,lastV,m,n)
            currentU, currentV = update_U_V(lastU,lastV,beta,currentX,currentW,currentZ)
            errorXW, errorXZ = converge(currentX,currentW,currentZ)
            print("error_XW:",errorXW)
            print("error_XZ:",errorXZ)
            lastW = currentW
            lastZ = currentZ 
            lastX = currentX 
            lastU = currentU 
            lastV = currentV                
        XSD_hat_arr = currentX.flatten()  # predicted results flatten         
        pos_test_id = test_all[fold_int]
        pos_test = known_sample[pos_test_id]
        pos_test_mask_list = np.zeros_like(A_orig_arr)
        pos_test_mask_list[pos_test] =1 
        pos_test_mask = pos_test_mask_list.reshape((m,n))
        neg_test_mask_candidate = get_negative_samples(pos_test_mask, drug_dissimmat)
        neg_test_mask = np.multiply(neg_test_mask_candidate, A_unknown_mask)
        neg_test = np.where(neg_test_mask.flatten() ==1)[0]                         
        pos_test_samples = XSD_hat_arr[pos_test]
        neg_test_samples = XSD_hat_arr[neg_test]       
        pos_labels = np.ones_like(pos_test_samples)
        neg_labels = np.zeros_like(neg_test_samples)        
        labels = np.hstack((pos_labels,neg_labels))
        scores = np.hstack((pos_test_samples,neg_test_samples))
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(labels,scores)
        print('TP:',TP)
        print('FP:',FP)
        print('FN:',FN)
        print('TN:',FN)
        print('fpr:',fpr)
        print('tpr:',tpr)
        print('auroc:',auroc)
        print('aupr:',aupr)
        print('f1_score:',f1_score)
        print('acc:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
        overall_auroc += auroc
        overall_aupr += aupr
        overall_f1 += f1_score
        overall_acc += accuracy
        overall_recall += recall
        overall_specificity +=specificity
        overall_precision += precision
    auroc_ = overall_auroc/10
    aupr_ = overall_aupr/10
    f1_ = overall_f1/10
    acc_ = overall_acc/10
    recall_ = overall_recall/10
    specificity_ = overall_specificity/10
    precision_ = overall_precision/10
    print('mean_auroc:',auroc_)
    print('mean_aupr:',aupr_)
        
        
        
        
        
        
        
        
        
        
        

     
        
        
        
        
        
        
        
    
