import sagemaker
from sagemaker import get_execution_role
import boto3
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
from datetime import timedelta
from datetime import date

sess = sagemaker.Session()

role = get_execution_role()
print(role)

bucket = 'carrefour-records'
print(bucket)

data_key = 'SeptoJanOrders.csv'
data_location = 's3://carrefour-records/SeptoJanOrders.csv'.format(bucket, data_key)
customer_loc = 's3://carrefour-records/orders-customer20190901-20200131.xlsx'.format(bucket, data_key)
product_loc = 's3://carrefour-records/orders-products20190901-20200131.csv'.format(bucket, data_key)

#SeptoJan = pd.read_csv(data_location)
customer = pd.read_excel(customer_loc)
products = pd.read_csv(product_loc)

#drop all the nulls
customer.dropna(inplace = True)
products.dropna(inplace = True)

#drops unnecessary colums
customer.drop(columns=['Revenue'], inplace = True)
products.drop(columns=['Month','Revenue'], inplace = True)

#drops duplicates
products.drop_duplicates(subset=['Order ID (evar17)'], inplace = True)

#merges data frames to get the full records of all transactions
df = pd.merge(customer,products, on='Order ID (evar17)')

#drops unnecessary column
df.drop(columns=['Order ID (evar17)'], inplace = True)

#pick out a particular customer
g = input("Enter customer id : ") 
cust = int(g)
Customer6976 = df.loc[df['Customer ID (evar5)'] == cust]
Customer6976.drop(columns=['Customer ID (evar5)'], inplace=True) # drop column customer id 
#Customer6976['Products'] = pd.to_numeric(Customer6976['Products'], downcast='float')

#chnage format to datetime
Customer6976['Month'] = pd.to_datetime(Customer6976['Month'])

#get the frequency for the rf model
#f stands for frequency
frequency = Customer6976.drop(columns=['Month'])
frequency['count'] = 1
frequency = frequency.groupby('Products')['count'].sum()

#create a new df for the purpose of calculating the recency
recency = Customer6976
recency['timestamp'] = "2020-1-1" #add a column with a time stamp
recency['timestamp'] = pd.to_datetime(recency['timestamp']) # channge the new column to date time
recency['recency'] = recency["timestamp"] - recency["Month"] # calculate the recency and pu it in a new column
recency.drop(columns=['timestamp'], inplace=True) # drop columns i no longer need 

##########################

#drops duplicates in recency
recency.sort_values('recency', ascending=True, inplace=True)
recency.drop_duplicates(subset='Products', keep='first', inplace = True)#drops the duplicates 

#merge for recency frequency 
rf = pd.merge(recency, frequency, on=['Products'])

#sorts the df
rf.sort_values('count', ascending=False, inplace = True)

#reset the axis
rf = rf.loc[:, rf.notnull().any(axis = 0)]

#change the type of a column to numeric
rf["Products"] = pd.to_numeric(rf["Products"])

#making two more dataframes one for september to december (the training data) and one for january (the testing data)
rfSept_to_Dec = rf[rf["Month"].isin(['2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01'])]
rfJan = rf[rf["Month"].isin(['2020-01-01'])]

######################

#label the probabality for every scenario
#sets for the most recent last month
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 4) & (rfSept_to_Dec['recency'] < pd.Timedelta(32,'D')), 'score'] = 'Most Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 3) & (rfSept_to_Dec['recency'] < pd.Timedelta(32,'D')), 'score'] = 'Very Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 2) & (rfSept_to_Dec['recency'] < pd.Timedelta(32,'D')), 'score'] = 'Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 1) & (rfSept_to_Dec['recency'] < pd.Timedelta(32,'D')), 'score'] = 'Maybe'
#sets for the items bought 2 months two months ago
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 4) & (rfSept_to_Dec['recency'] == pd.Timedelta(61,'D')), 'score'] = 'Very Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 3) & (rfSept_to_Dec['recency'] == pd.Timedelta(61,'D')), 'score'] = 'Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 2) & (rfSept_to_Dec['recency'] == pd.Timedelta(61,'D')), 'score'] = 'Maybe'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 1) & (rfSept_to_Dec['recency'] == pd.Timedelta(61,'D')), 'score'] = 'UnLikely'
#sets for the items bought 3 months two months ago
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 4) & (rfSept_to_Dec['recency'] == pd.Timedelta(92,'D')), 'score'] = 'Likely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 3) & (rfSept_to_Dec['recency'] == pd.Timedelta(92,'D')), 'score'] = 'Maybe'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 2) & (rfSept_to_Dec['recency'] == pd.Timedelta(92,'D')), 'score'] = 'Unlikely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 1) & (rfSept_to_Dec['recency'] == pd.Timedelta(92,'D')), 'score'] = 'UnLikely'
#sets for the items bought 4 months two months ago
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 4) & (rfSept_to_Dec['recency'] == pd.Timedelta(122,'D')), 'score'] = 'Maybe'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 3) & (rfSept_to_Dec['recency'] == pd.Timedelta(122,'D')), 'score'] = 'Unlikely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 2) & (rfSept_to_Dec['recency'] == pd.Timedelta(122,'D')), 'score'] = 'Unlikely'
rfSept_to_Dec.loc[(rfSept_to_Dec['count'] == 1) & (rfSept_to_Dec['recency'] == pd.Timedelta(122,'D')), 'score'] = 'UnLikely'

########################

########################

#Assign the different tags to dataframes
MostLikely = rfSept_to_Dec.loc[rfSept_to_Dec['score'] == 'Most Likely']
VeryLikely = rfSept_to_Dec.loc[rfSept_to_Dec['score'] == 'Very Likely']
Likely = rfSept_to_Dec.loc[rfSept_to_Dec['score'] == 'Likely']
MVL = rfSept_to_Dec[rfSept_to_Dec['score'].isin(['Most Likely', 'Very Likely', 'Likely'])]

#######################