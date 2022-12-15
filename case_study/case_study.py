# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 00:15:30 2022

@author: Jiani Ma
"""

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt




path = "D:/11_DTI_work/revised_pic/figures/P0DTC2_CTD.xlsx"

data = pd.read_excel(path,index_col="protein").values[0] 


sns.heatmap(data) 