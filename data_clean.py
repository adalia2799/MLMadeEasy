import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import os 
import base64
def main():
    st.title("MLMadeEasy")
    st.sidebar.title("MLMaDeEasy")
    st.markdown("Fit your Machine Learning Model with Easy")
    st.sidebar.markdown("Data Cleaning")
    filename = st.sidebar.file_uploader("Choose file", type="csv")
    if filename is not None:
        df = pd.read_csv(filename)
        column =df.columns
        selected_column = st.sidebar.selectbox("Select column",column)
        #column view 
        if st.sidebar.checkbox("View columns details",False):
            st.subheader("Basic Statistics")
            st.table(df[selected_column].describe())
            missing_value = int(df[selected_column].isnull().sum())
            size = int(len(df[selected_column]))
            percent_missing = 100*(missing_value/size)
            st.write('Missing value Percantage =',percent_missing,'%')
        #data imputation 
        st.sidebar.subheader("Data Imputation")
        
        method_of_clean = st.sidebar.selectbox("Choose method to Impute Missing",("Mean","Median","Mode","HardCode"))
        if st.sidebar.checkbox("Choose to Apply",False):
            if method_of_clean=="Drop":
                df = df.dropna()
                st.warning("All Rows with NULL value Deleted")
            elif method_of_clean=="Mean":
                #select all column with continous value
                numeric_col =[]
                for col in df.columns:
                    if(len(df[col].value_counts())>20):
                        numeric_col.append(col)
                column_to_clean = st.sidebar.multiselect('Select column to impute by mean',numeric_col)
                for col in column_to_clean:
                    if(df[col].isnull().sum()==0):
                        st.warning("There is no missing value in "+col)
                        continue
                    mean_value = df[col].mean()
                    df[col] = df[col].fillna(mean_value)
                    st.warning(col+" is imputed with "+str(mean_value))
                    if st.checkbox("Show mean Imputed col of "+col,False):
                        st.write(df[col])
            elif method_of_clean=="Median":
                numeric_col =[]
                for col in df.columns:
                    if(len(df[col].value_counts())>20):
                        numeric_col.append(col)
                column_to_clean = st.sidebar.multiselect('Select column to impute by mode',numeric_col)
                for col in column_to_clean:
                    if(df[col].isnull().sum()==0):
                        st.warning("There is no missing value in "+col)
                        continue
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    st.warning(col+" is imputed with "+str(median_value))
                    if st.checkbox("Show median Imputed col of "+col,False):
                        st.write(df[col])
            
            elif method_of_clean=="Mode":
                categorical_col =[]
                for col in df.columns:
                    if(len(df[col].value_counts())<=20):
                        categorical_col.append(col)
                column_to_clean = st.sidebar.multiselect('Select column to impute by mode',categorical_col)
                for col in column_to_clean:
                    if(df[col].isnull().sum()==0):
                        st.warning("There is no missing value in "+col)
                        continue
                    mode_value = df[col].mode()
                    df[col] = df[col].fillna(mode_value)
                    st.warning(col+" is imputed with "+str(mode_value))
                    if st.checkbox("Show mode Imputed col of "+col,False):
                        st.write(df[col])
            #option for download file 
            
        #drop columns in pandas 
        st.sidebar.subheader("Drop unwanted columns")
        if st.sidebar.checkbox("Drop column",False):
            column_to_drop = st.sidebar.multiselect("Select column to drop",df.columns)
            df = df.drop(column_to_drop,axis=1)
            str1 =','.join(column_to_drop)
            st.warning(str1+" Dropped")
            st.write(df)
        st.sidebar.subheader("One hot encoding")
        if st.sidebar.checkbox("Do one-hot encoding",False):
            df = pd.get_dummies(df)
            st.write(df)
    
        if(st.sidebar.button("Download file","download")):
            st.subheader("Download the processed file")
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)

            




if __name__=="__main__":
    main()