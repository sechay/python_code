
# coding: utf-8

# In[6]:

import mysql.connector as msq
import pandas as pd
import numpy as np
import sqlalchemy as sql
import pymysql


# In[7]:

df = pd.read_csv("/Users/naveen/Documents/python_code/Test/dataset-Sg/singapore-citizens.csv")


# In[8]:

database_connection = sql.create_engine('mysql+pymysql://naveen:naveen@192.168.0.112:3307/NAVEEN')
df.to_sql(con=database_connection, name='sg_citizen', if_exists='replace')
database_connection.close()


# In[11]:

connection = msq.connect(host = '192.168.0.112',
                         port = '3307',                    
                       user = 'naveen',
                       password = 'naveen',
                        database = 'NAVEEN')
cursor = connection.cursor()
cursor.execute("select * from sg_citizen")
myresult = cursor.fetchall()
for x in myresult:
    print(x)
connection.close()


# In[ ]:



