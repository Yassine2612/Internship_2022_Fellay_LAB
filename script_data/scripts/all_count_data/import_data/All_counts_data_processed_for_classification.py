import pandas as pd
import numpy as np


# As we have over 37000 genes , we spread them into 3 excel sheets 
df_1 = pd.read_excel('../../../data/all_normalised_gene_counts.xlsx' , sheet_name=0)
df_2 = pd.read_excel('../../../data/all_normalised_gene_counts.xlsx' , sheet_name=1)
df_3 = pd.read_excel('../../../data/all_normalised_gene_counts.xlsx' , sheet_name=2)

#replace healthy-critical in LABEL with 0-1

df_3.LABEL = df_3.LABEL.replace('critical',1)
df_3.LABEL = df_3.LABEL.replace('healthy',0)


frames = [df_1, df_2 ,df_3 ]
  
df = pd.concat(frames ,axis =1)

#Remove the first column gene_id 
df = df.drop(df.columns[0], axis=1)

#Each gene is an attribute of the data
attributeNames = np.asarray(df.columns)

raw_data = df.values  

cols = range(0, 37194) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the excel (gene names )
attributeNames = np.asarray(df.columns[cols])

# We extract the sample label (sample_group)
classLabels = raw_data[:,-1] # -1 takes the last column

classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)

print("the number of rows is equal", N , "to the number of samples ")
print("the number of columns is equal", M , "to the number of genes ")
print("We have ",C, "different sample classes ")
