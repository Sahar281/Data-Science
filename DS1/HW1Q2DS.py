
import pandas as pd
df=pd.read_csv('customerData.csv')
#Q2
print (df.describe(include='all'))
#Q2a
print (df.shape)
#Q2b
# numerical columns: custid, income, num_vehicles, age
# categorial columns: gender, is_employed, marital_stat, health_ins, housing_type, recent_move, state_of_res

#Q2c
# hamadadim be custid hem misparey zihuy shel anashim, eynam madadim statistim ve lahen lo ozer limdod et hanetunim halalu,
#hameida she itkabel lo ihiye shimushi.
#Q3
df_sub=df.iloc[::10,::2]
print (df_sub)
#Q4
#Q4a
print (df.shape)
#Q4b
print (df.size)
#Q4c
# efshar lehashev et size beemzaut hapkuda shape batsura habaa:
print(df.shape[0]*df.shape[1])
#Q5
df_sub_ages=df[(df['age']>38) & (df['age']<=50)]
print (df_sub_ages)
#Q6
#Q6a
df_q6=df[df['age']>50].select_dtypes(include='number')
print (df_q6)
#Q6b
df_q6_2=df[df['age']>50].iloc[:,[0,3,8,9]]
print (df_q6_2)
#Q7
df_7=df_q6.loc[:,['age']].head(100)
print (df_7)
#Q8
print (df[(df['age']<18)&(df['marital_stat'].isin(['married','Divorced/Separated']))].iloc[:,[0]])
#Q9
#Q9a
print (df[(df['income']>16000)&(df['state_of_res']=='Washington')]['age'].mean())
#Q9b
print (df[(df['income']>16000)&(df['state_of_res']=='Washington')]['age'].max())
#Q9c
print (df[(df['income']>16000)&(df['state_of_res']=='Washington')]['income'].min())
#Q9d
print (df[(df['income']>16000)&(df['state_of_res']=='Washington')].shape[1])
#Q10
df10=df.groupby(['gender','housing_type'])
#Q10a
print (df10.size()['F'].idxmax())
#Q10b
print (df10.size()['M'].idxmax())
