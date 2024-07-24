#Data-Preprocessing 
# Follow following steps for data pre-processing
#1. Data Cleaning
#     a.Low-expressed genes were filtered out of the transcriptome data [log2 (RSEM + 1) < 0.1 in the 90% sample]
##    b.Checking of any missing value. 
##    c.Data Normalization


#Importing the required lib

#Filling of mean values

import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
df1=pd.read_csv("G2_G3_combineddata_F3.csv") 
df1.shape


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df1.iloc[:, 1:])
df1.iloc[:, 1:] = imputer.transform(df1.iloc[:, 1:])

df1.to_csv("G2_G3_combineddata_F3samplesmean.csv")




