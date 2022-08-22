import pandas as pd
import numpy as np


df = pd.read_excel('../../data/small_normalised_gene_counts.xlsx')


#replace healthy-critical in LABEL with 0-1
df.LABEL = df.LABEL.replace('critical',1)
df.LABEL = df.LABEL.replace('healthy',0)

#Remove the first column gene_id 
df = df.drop(df.columns[0], axis=1)
#Each gene is an attribute of the data
attributeNames = np.asarray(df.columns)
raw_data = df.values  

cols = range(0, 200) 
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


