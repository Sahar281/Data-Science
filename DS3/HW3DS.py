
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Q1

df = pd.read_csv('Bridges.csv')
df.info()
# provides num of rows,all columns names, data types, and how much null values in each column
print(df.describe(include='all'))
# describe command provides mean, median and many more useful info
print(df.isna())
# # isna command show rows and cols and every place null occur true value shown else false
print(df.isna().sum())
# # isna.sum.sum summarise each col and num of null values
print(df)

# Q2

df = df.dropna(subset=['T-OR-D', 'MATERIAL'])
# 2.a cols 'T-OR-D','MATERIAL' have under 10 percent of null values then were dropping the whole row
print(df.isna().sum())

df['SPAN'] = df['SPAN'].replace(to_replace=np.NaN, value='Not measured')
# 2.b SPAN col dtype = Object will be treated as missing data in categorical data

print(df[['LENGTH', 'LANES']].describe())

# using summary statistics to spot problems in numeric data & getting median values
df['LANES'] = df['LANES'].replace(to_replace=np.nan, value=df.LANES.mean())

# treating missing values with mean
df['LENGTH'] = df['LENGTH'].replace(to_replace=np.nan, value=df.LENGTH.median())

# treating missing values with median dou to significant outliers
# 2.c treating missing data in numeric data

print(df.isna().sum())
# printing num of nulls in each coll

# Q3

plt.hist(df['LENGTH'], bins=10)
plt.show()
# printing histogram of length column
# the optional transformations are normalization and Log transformation

df['LENGTHlog'] = df['LENGTH'].mask(df['LENGTH'] < 1, 1)
df['LENGTHlog'] = np.log10(df['LENGTHlog'])
# creating new column named LENGTHlog using mask command on LENGTH column then using log transformation on LENGTHlog

df['LENGTHlogNorm'] = stats.zscore(df['LENGTHlog'])
# creating new column named LENGTHlogNorm that normalize LENGTHlog
plt.hist(df['LENGTHlogNorm'], bins=10)
plt.show()
dfnum = df.select_dtypes(include='number')
# creating new data frame named dfnum including all numbers datatypes and normalize them
dfnum = dfnum.apply(stats.zscore)
# # normalize all columns of dfnum
dfnum = dfnum.rename(columns={'ERECTED': 'ERECTEDnorm', 'LENGTH': 'LENGTHnorm', 'LANES': 'LANESnorm', 'LENGTHlog': 'LENGTHlogNORM', 'LENGTHlogNorm': 'LENGTHLOGnorm'})
# rename columns
df = pd.concat([df, dfnum], axis=1)

# concatinate dfnum with df

# Q4a

bins = [df.ERECTED.min()-1, 1901, 1961, df.ERECTED.max()+1]
# defining bins gaps
labels = ['very old', 'old', 'modern']
# labeling the gaps
df['ERECTEDrange'] = pd.cut(df['ERECTED'], bins=bins, labels=labels)
# creating a new column named ERECTEDrange & using pandas cut func on ERECTED column
print(df[['ERECTEDrange', 'ERECTED']])
# printing columns themselves

# Q4b

MATERIALcols = pd.get_dummies(df['MATERIAL'], prefix='MATERIALType-')
# creating new col for each value type at MATERIAL col
TORDcols = pd.get_dummies(df['T-OR-D'], prefix='T-OR-DType-')
# creating new col for each value type at T-OR-D col

df = pd.concat([df, MATERIALcols, TORDcols], axis=1)
# concatenate new cols with data frame vertically
df = df.drop(['MATERIAL', 'T-OR-D'], axis=1)
# dropping previous cols

dfnum2 = df.drop(['IDENTIF', 'RIVER', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'SPAN', 'LENGTHlog', 'LENGTHlogNorm', 'ERECTEDnorm', 'LENGTHnorm', 'LENGTHlogNORM'], axis=1)
# droping irrelevant cols and duplicated cols
print(dfnum2.info())