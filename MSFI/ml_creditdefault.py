
#Description: Classification model for Micro Finance Credit Default prediction
#Version: 1.0
#Author: Naveen Srinivasan

#-------------------------------------------------------------------------------
#pymysql to connect to mysql instance running on RDS
#pandas to be able to import the dataset as a frame
#install all the packages required using pip
#-------------------------------------------------------------------------------



import pip
pip.main(['install', 'pymysql'])
pip.main(['install', 'sklearn'])
pip.main(['install', 'numpy'])


#import the required packages
import pymysql
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sys import argv



#-------------------------------------------------------------------------------
"""**Connecting to RDS Mysql to read it into dataframe df**"""
#-------------------------------------------------------------------------------
def main(argv):

    connection = pymysql.connect(host='database-1.crfkl2juooug.ap-southeast-1.rds.amazonaws.com',
                             user='prithvi',
                             password='NannaMooda',                             
                             db='micfinml',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


        
    df = pd.read_sql('SELECT * FROM micfinml.creditDefault_nonbias', con=connection)
 
 
    connection.close()

    #-------------------------------------------------------------------------------
    """Choose the significant predictors. Drop ID and call the new dataframe df_without_id"""
    #-------------------------------------------------------------------------------

    df_without_id = df.drop('ID', axis=1)

    """**Now, split our DataFrame into input data and target**"""

    data = df_without_id[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'RS_0', 'RS_2',
       'RS_3', 'RS_4', 'RS_5', 'RS_6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

    target = df_without_id[['p_default']]

    #-------------------------------------------------------------------------------
    """**Defining the Train and Test dataset**
    --- train_test_split function splits random train and test subsets
    """
    #-------------------------------------------------------------------------------


    train_data, test_data, train_target, test_target = train_test_split(data, target,random_state=42)

    #-------------------------------------------------------------------------------
    """**Train a Decision tree model**"""
    #-------------------------------------------------------------------------------


    dtc = DecisionTreeClassifier(random_state=42, max_leaf_nodes=10, max_depth=5)
    dtc.fit(train_data, train_target)

    predict_Default(argv)

def predict_Default(argv):
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q = argv
    x2 = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q]
    A = np.array(x2)
    B = np.reshape(A,(1,17))
  
    pred2 = dtc.predict(B)

    return(pred2);
