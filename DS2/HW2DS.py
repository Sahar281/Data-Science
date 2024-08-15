import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clubmed_HW2.csv')
# Q1
#Q1a
plt.hist(df['age'])
plt.xlabel('age')
plt.ylabel('population')
plt.title('ages histogram')
plt.show()
#Q1b
plt.hist(df['age'],bins=50)
plt.xlabel('age')
plt.ylabel('population')
plt.title('ages histogram')
plt.show()
plt.hist(df['age'],bins=5)
plt.xlabel('age')
plt.ylabel('population')
plt.title('ages histogram')
plt.show()
#Nitan lilmod she kehol she maalim et godel ha bins hahistograma tihiye meduyeket yoter.
# Q2
member=df['club_member'].value_counts()
plt.bar(member.index.map({True:'true',False:'false'}),member.values,color=('green','pink'))
plt.xlabel('club member')
plt.ylabel('population')
plt.show()

# Q3
#Q3a
crosstab=pd.crosstab(index=df['gender'],columns=df['status'])
crosstab.plot.bar(stacked=True)
plt.show()
#Q3b+c
crosstab2=pd.crosstab(index=df['status'],columns=df['gender'])
crosstab2.plot.bar(stacked=True)
plt.show()
#Q3d
# ahuz ha gvarim ha gavoa beyoter bestatus couple. hi gdola be hehrah mikol sh'ar ha statusim ha mishpahtiim.
#Q3e
# ha status ha shahiah bekerev ha nashim hu couple.
#Q3f
# ahuz a nashim she lo nesuot me klal ha nashim hino 27/92*100 klomar 29.347%.
#Q3g
# ahuz a gvarim ha ravakim me klal ha ravakim hino 27/54*100 klomar 50%.

# Q5
#Q5a
crosstab3=pd.crosstab(index=df['region'],columns=df['accomodation'])
crosstab3.plot.bar(stacked=True)
plt.xlabel('ragion')
plt.ylabel('accomodation')
plt.show()
#Q5b
# nitan lilmod me ha graph she Eilat ze a makom ayahid she lokhim bo deluxe villa yoter maasher ha sugim ha aherim
# shel ahadarim.
#Q5c
crosstab4=pd.crosstab(index=df['gender'],columns=df['club_member'])
crosstab4.plot.bar(stacked=True)
plt.show()
# le status yesh kesher megamati im gender.
# Q6
plt.scatter(df.age,df.minibar)
plt.xlabel('age')
plt.ylabel('minibar expences')
plt.title('minibar expenses among the ages')
plt.show()
# Q7
#Q7a
df_sub=df[df['room_price'].notna()]
q1=df_sub['room_price'].quantile(0.25)
q3=df_sub['room_price'].quantile(0.75)
iqr=q3-q1
print (iqr)
print(df_sub.room_price.std())
#Q7b
med=df_sub.room_price.median()
print(df_sub[df_sub['room_price']<=med]['room_price'].size)
#ze toem et hagdarat a hetzion.
#Q7c
mean=df_sub.room_price.mean()
std=df_sub.room_price.std()
low_std=mean-std
high_std=mean+std
plt.hist(df_sub['room_price'])
plt.axvline(x=mean,color='red')
plt.axvline(x=low_std,color='black')
plt.axvline(x=high_std,color='black')
plt.show()
# hitpalgut a-simetrit yemanit.kayamim arahim kitsoniyim, aval apaamon tzar.
# Q8
#Q8a
df.boxplot(column =['age'], by ='ranking',grid = False)
plt.show()
#ha IQR ahi gadol mitpares be ranking 2
#Q8b
q1a=df['age'].quantile(0.25)
q3a=df['age'].quantile(0.75)
iqr=q3a-q1a
outliers=q3a+1.5*iqr
df.boxplot(column =['age'], by ='ranking',grid = False)
plt.axhline(y=outliers,color='red')
plt.show()
# Q9
df.boxplot(column =['age'], by ='visits5years',grid = False)
plt.show()
#Q9a
#7 bikurim
#Q9b
print(df[df['visits5years']==7].age.values)
#Q9c
df.boxplot(column =['room_price'], by ='visits5years',grid = False)
plt.show()
# Medubar be adam boded, en po shum davar meyatseg. behol ofen amehir shehu meshalem zol yoter mi kol sh'ar
# hamehirim hamemutsaim haaherim.(kol ele she lo bikru 7 peamim)
# Q10
df.boxplot(column =['total_expenditure'], by ='ranking',grid = False)
plt.show()
#lo nitan lirot kesher muvhak. nikah ledugma et hameduragim 1 (lowest) ve meduragim 5(very high). memutsa ha hotsaot
# shel meduragim 1 gavoa yoter mi memutsa ha hotsaot shel medoragim 5 velahen en kesher muvhak ben ranking le bizbuzim.