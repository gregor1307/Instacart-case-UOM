

# In[1]:


# For data manipulation
import pandas as pd 


# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[2]:


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi({"username":"gregorchatzi","key":"bfea27556eac16f9a946432c3d70ea89"})
api.authenticate()
files = api.competition_download_files('instacart-market-basket-analysis')


# In[3]:


import zipfile
with zipfile.ZipFile('instacart-market-basket-analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('./input')


# In[4]:


import os
working_directory = os.getcwd()+'/input'
os.chdir(working_directory)
for file in os.listdir(working_directory):   # get the list of files
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory


# ## 1.2 Load data from the CSV files
# Instacart provides 6 CSV files, which we have to load into Python. Towards this end, we use the .read_csv() function, which is included in the Pandas package. Reading in data with the .read_csv( ) function returns a DataFrame.

# In[5]:


orders = pd.read_csv('../input/orders.csv')
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')


# This step results in the following DataFrames:
# * <b>orders</b>: This table includes all orders, namely prior, train, and test. It has single primary key (<b>order_id</b>).
# * <b>order_products_train</b>: This table includes training orders. It has a composite primary key (<b>order_id and product_id</b>) and indicates whether a product in an order is a reorder or not (through the reordered variable).
# * <b>order_products_prior </b>: This table includes prior orders. It has a composite primary key (<b>order_id and product_id</b>) and indicates whether a product in an order is a reorder or not (through the reordered variable).
# * <b>products</b>: This table includes all products. It has a single primary key (<b>product_id</b>)
# * <b>aisles</b>: This table includes all aisles. It has a single primary key (<b>aisle_id</b>)
# * <b>departments</b>: This table includes all departments. It has a single primary key (<b>department_id</b>)

# If you want to reduce the execution time of this Kernel you can use the following piece of code by uncomment it. This will trim the orders DataFrame and will keep a 10% random sample of the users. You can use this for experimentation.

# In[6]:



#### Remove triple quotes to trim your dataset and experiment with your data
# COMMANDS FOR CODING TESTING - Get 10% of users 
###orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.1, random_state=25))] 


# We now use the .head( ) method in order to visualise the first 10 rows of these tables. Click the Output button below to see the tables.

# In[7]:


orders.head()


# In[8]:


order_products_train.head()


# In[9]:


order_products_prior.head()


# In[10]:


products.head()


# In[11]:


aisles.head()


# In[12]:


departments.head()


# ## 1.3 Reshape data
# We transform the data in order to facilitate their further analysis. First, we convert character variables into categories so we can use them in the creation of the model. In Python, a categorical variable is called category and has a fixed number of different values.

# In[13]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')



# In[14]:


#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()



# In[15]:


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1
op.head(15)


# In[16]:


op5 = op[op.order_number_back <= 5]
op5.head()


#
# In[17]:


user = op.groupby('user_id')['order_number'].max().to_frame('user_t_orders') #
user = user.reset_index()
user.head()


# In[18]:


u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio') #
u_reorder = u_reorder.reset_index()
u_reorder.head()


# In[19]:


u_reorder5 = op5.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio_5') #
u_reorder5 = u_reorder5.reset_index()
u_reorder5.head()


# In[20]:


user=user.merge(u_reorder,on='user_id',how='left')#
del u_reorder
gc.collect()

user.head()


# In[21]:


user=user.merge(u_reorder5,on='user_id',how='left')#
del u_reorder5
gc.collect()

user.head()


# In[22]:


days = op.groupby('user_id')['days_since_prior_order'].mean().to_frame('mean_days')
days = days.reset_index()
days.head()


# In[23]:


days5 = op5.groupby('user_id')['days_since_prior_order'].mean().to_frame('mean_days_5')
days5 = days5.reset_index()
days5.head()


# In[24]:


user = user.merge(days, on='user_id', how='left')
del days
gc.collect()
user.head()


# In[25]:


user = user.merge(days5, on='user_id', how='left')
del days5
gc.collect()
user.head()


# In[26]:


user_max = op.groupby(['user_id','order_id'])['add_to_cart_order'].max().to_frame('max_basket')
user_max.head()


# In[27]:


user_max_ratio = user_max.groupby('user_id')['max_basket'].mean().to_frame('mean_basket')
user_max_ratio = user_max_ratio.reset_index()
user_max_ratio.head()


# In[28]:


user = user.merge(user_max_ratio, on='user_id', how='left')
del user_max_ratio
gc.collect()
user.head()


# In[29]:


user_max5 = op5.groupby(['user_id','order_id'])['add_to_cart_order'].max().to_frame('max_basket5')
user_max5.head()


# In[30]:


user_max_ratio5 = user_max5.groupby('user_id')['max_basket5'].mean().to_frame('mean_basket5')
user_max_ratio5 = user_max_ratio5.reset_index()
user_max_ratio5.head()


# In[31]:


user = user.merge(user_max_ratio5, on='user_id', how='left')
del user_max_ratio5
gc.collect()
user.head()


# In[32]:


del  user_max, user_max5
gc.collect()
user.head()


# # **Products**

# In[33]:


# Create distinct groups for each product, count the orders, save the result for each product to a new DataFrame  
prd = op.groupby('product_id')['order_id'].count().to_frame('prd_t_purchases') #
prd = prd.reset_index()
prd.head()


# In[34]:


# Reset the index of the DF so to bring product_id rom index to column (pre-requisite for step 2.4)
prd5 = op5.groupby('product_id')['order_id'].count().to_frame('prd_t_purchases5') #
prd5 = prd5.reset_index()
prd5.head()


# In[35]:


prd = prd.merge(prd5, on='product_id', how='left')

#delete the reorder DataFrame
del prd5
gc.collect()

prd.head()


# In[36]:



p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)#####
p_reorder.head()


# In[37]:


p_reorder = op.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()
p_reorder.head()


# In[38]:


#Merge the prd DataFrame with reorder
prd = prd.merge(p_reorder, on='product_id', how='left')

#delete the reorder DataFrame
del p_reorder
gc.collect()

prd.head()


# In[39]:


prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(0) #
prd.head()


# In[40]:


p_reorder5 = op5.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio5')
p_reorder5 = p_reorder5.reset_index()
p_reorder5.head()


# In[41]:


prd = prd.merge(p_reorder5, on='product_id', how='left')

#delete the reorder DataFrame
del p_reorder5
gc.collect()
prd['p_reorder_ratio5'] = prd['p_reorder_ratio5'].fillna(0)
prd.head()


# In[42]:


aop = op.groupby('product_id')['add_to_cart_order'].mean().to_frame("aop_mean")
aop = aop.reset_index()
aop.head()


# In[43]:


prd = prd.merge(aop, on='product_id', how='left')
del aop
gc.collect()
prd.head()


# In[44]:


aop5 = op5.groupby('product_id')['add_to_cart_order'].mean().to_frame("aop_mean5")
aop5 = aop5.reset_index()
aop5.head()


# In[45]:


prd = prd.merge(aop5, on='product_id', how='left')
del aop5
gc.collect()
prd.head()


# In[46]:


ap5=products.merge(op5,on='product_id',how='right')
ap5=ap5.groupby(['user_id','aisle_id'])['order_id'].count().to_frame('times_aisle')
ap5.head()


# In[47]:


ap5_sum=ap5.groupby('user_id')['times_aisle'].sum().to_frame('t_times_aisle')
ap5_sum.head()


# In[48]:


ap5=ap5.merge(ap5_sum,on='user_id',how='left')
ap5.head()


# In[49]:


ap5['freq_aisle_per_costumer']=ap5.times_aisle/ap5.t_times_aisle
ap5=ap5.drop(['times_aisle','t_times_aisle'],axis=1)
ap5.head()


# In[50]:


dp5=products.merge(op5,on='product_id',how='right')
dp5=dp5.groupby(['user_id','department_id'])['order_id'].count().to_frame('times_dep')
dp5.head()


# In[51]:


dp5_sum=dp5.groupby('user_id')['times_dep'].sum().to_frame('t_times_dep')
dp5_sum.head()


# In[52]:


dp5=dp5.merge(dp5_sum,on='user_id',how='left')
dp5.head()


# In[53]:


dp5['freq_dep_per_costumer']=dp5.times_dep/dp5.t_times_dep
dp5=dp5.drop(['times_dep','t_times_dep'],axis=1)
dp5.head()


# In[54]:


user=user.merge(ap5,on='user_id',how='left')
user.head()


# In[55]:


user=user.merge(dp5,on='user_id',how='left')
user.head()


# In[56]:


del ap5,ap5_sum,dp5,dp5_sum
gc.collect()


# In[57]:


# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 
uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_t_bought') #
uxp = uxp.reset_index()
uxp.head()


# In[58]:


times = op.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# In[59]:


total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders') #
total_orders.head()


# In[60]:


first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# In[61]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# In[62]:


# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# In[63]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# In[64]:


#Remove temporary DataFrames
del [times, first_order_no, span]


# In[65]:


uxp_ratio['uxp_reorder_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D ##
uxp_ratio.head()


# In[66]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# In[67]:


uxp = uxp.drop(['total_orders', 'first_order_number', 'Order_Range_D', 'Times_Bought_N'], axis=1)
uxp.head()


# In[68]:


last_five = op5.groupby(['user_id','product_id'])['order_id'].count().to_frame('times_last5')
last_five.head(10)


# In[69]:


uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')
uxp['times_last5'] = uxp['times_last5'].fillna(0)
del last_five
uxp.head()


# In[70]:


del op
gc.collect()



# In[71]:


#Merge uxp features with the user features
#Store the results on a new DataFrame
data = uxp.merge(user, on='user_id', how='left')
data.head()


# In[72]:


del uxp, user
gc.collect()


# In[74]:


#Merge uxp & user features (the new DataFrame) with prd features
data = data.merge(prd, on='product_id', how='left') #
data.head()


# In[ ]:


del prd
gc.collect()



# In[ ]:


## First approach:
# In two steps keep only the future orders from all customers: train & test 
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)



# In[ ]:


# bring the info of the future orders to data DF
data = data.merge(orders_future, on='user_id', how='left')
data.head(10)



# In[ ]:


#Keep only the customers who we know what they bought in their future order
data_train = data[data.eval_set=='train'] #
data_train.head()




# In[ ]:


#Get from order_products_train all the products that the train users bought bought in their future order
data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head(15)



# In[ ]:


#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.
data_train['reordered'] = data_train['reordered'].fillna(0)
data_train.head(15)


# In[ ]:


#We set user_id and product_id as the index of the DF
data_train = data_train.set_index(['user_id', 'product_id'])
data_train.head(15)


# In[ ]:


#We remove all non-predictor variables
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head(15)



# In[ ]:


#Keep only the future orders from customers who are labelled as test
data_test = data[data.eval_set=='test'] #
data_test.head()


# In[ ]:


#We set user_id and product_id as the index of the DF
data_test = data_test.set_index(['user_id', 'product_id']) #
data_test.head()


# In[ ]:


#We remove all non-predictor variables
data_test = data_test.drop(['eval_set','order_id'], axis=1)
#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable
data_test.head()



# In[ ]:


import xgboost as xgb


X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered


parameters = {'eval_metric':'logloss', 
              'max_depth': 5, 
              'colsample_bytree': 0.4,
              'subsample': 0.75,
             }


xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)


model = xgbc.fit(X_train, y_train)

xgb.plot_importance(model)


# In[ ]:


model.get_xgb_params()


# In[ ]:



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

paramGrid = {"max_depth":[5,10],
            "colsample_bytree":[0.3,0.4,0.5],
            "n_estimators":[50,100]}  

xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',num_boost_round=10,sampling_method='gradient_based', gpu_id=1, tree_method = 'gpu_hist')


gridsearch = GridSearchCV(xgbc, paramGrid, cv=3, verbose=2, n_jobs=1)


model = gridsearch.fit(X_train, y_train)


# Print the best parameters
print("The best parameters are: /n",  gridsearch.best_params_)

# Store the model for prediction (chapter 5)
model = gridsearch.best_estimator_


# In[ ]:



xgb.plot_importance(model)


# In[ ]:


model.get_params()


# In[ ]:


del [orders_future,X_train,y_train]
gc.collect()




# In[ ]:


'''
# Predict values for test data with our model from chapter 5 - the results are saved as a Python array
test_pred = model.predict(data_test).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array
'''


# In[ ]:


## OR set a custom threshold (in this problem, 0.21 yields the best prediction)
test_pred = (model.predict_proba(data_test)[:,1] >= 0.21).astype(int)
test_pred[0:20] #display the first 20 predictions of the numpy array


# In[ ]:


#Save the prediction in a new column in the data_test DF
data_test['prediction'] = test_pred
data_test.head()


# In[ ]:


#Reset the index
final = data_test.reset_index()
#Keep only the required columns to create our submission file (Chapter 6)
final = final[['product_id', 'user_id', 'prediction']]

gc.collect()
final.head()



# In[ ]:


orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
orders_test.head()


# We merge it with our predictions (from chapter 5) using a left join:
# <img src="https://i.imgur.com/KJubu0v.jpg" width="400">

# In[ ]:


final = final.merge(orders_test, on='user_id', how='left')
final.head()


# In[ ]:


#remove user_id column
final = final.drop('user_id', axis=1)
#convert product_id as integer
final['product_id'] = final.product_id.astype(int)

#Remove all unnecessary objects
del orders
del orders_test
gc.collect()

final.head()

# In[ ]:


d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in final.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()

#We now check how the dictionary were populated (open hidden output)
d


# In[ ]:


#Convert the dictionary into a DataFrame
sub = pd.DataFrame.from_dict(d, orient='index')

#Reset index
sub.reset_index(inplace=True)
#Set column names
sub.columns = ['order_id', 'products']

sub.head()


# **The submission file should have 75.000 predictions to be submitted in the competition**

# In[ ]:


#Check if sub file has 75000 predictions
sub.shape[0]


# The DataFrame can now be converted to .csv file. Pandas can export a DataFrame to a .csv file with the .to_csv( ) function.

# In[ ]:


sub.to_csv('sub.csv', index=False)

# In[ ]:




