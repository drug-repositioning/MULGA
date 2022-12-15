# -*- coding: utf-8 -*-

"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from numpy import linalg as LA
import matplotlib.pyplot as plt

def read_data(data_folder,drug_sim_path,target_sim_path,DTI_path):
    SR = pd.read_excel(os.path.join(data_folder, drug_sim_path),header=None).values
    SD = pd.read_excel(os.path.join(data_folder, target_sim_path),header=None).values
    A_orig = pd.read_excel(os.path.join(data_folder, DTI_path),header=None).values 
    A_orig_arr = A_orig.flatten()
    known_sample = np.nonzero(A_orig_arr)[0]   
    return SR,SD,A_orig,A_orig_arr,known_sample


       


    






