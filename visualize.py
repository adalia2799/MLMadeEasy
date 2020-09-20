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
import seaborn as sns
from bokeh.models.widgets import ColorPicker

def main():
    st.title("MLMadeEasy")
    st.sidebar.title("MLMaDeEasy")
    st.markdown("Fit your Machine Learning Model with Easy")
    st.sidebar.markdown("Exploratory Data Analysis")
    filename = st.sidebar.file_uploader("Choose file", type="csv")
    if filename is not None:
        df = pd.read_csv(filename)
        column_list = df.columns
        categorical=[]
        continous =[]
        for i in column_list:
            length = len(df[i].value_counts())
            if(length<20):
                categorical.append(i)
            else:
                continous.append(i)
        #show head of the data 
        if(st.sidebar.checkbox("show head",False)):
            head =df.head()
            st.subheader("head of the dataset")
            st.table(df.head())
        if(st.sidebar.checkbox("show tail",False)):
            tail =df.tail()
            st.subheader("Tail of the data")
            st.table(tail)
        if(st.sidebar.checkbox("Descriptive Stats",False)):
            desc =df.describe()
            st.subheader("Descriptive Stats")
            st.table(desc)
        if st.sidebar.checkbox("Explore continous data",False):
            #univariate plot
            if(len(continous)==0):
                st.warning("There is no continous data to analysis")
            else:
                if st.sidebar.checkbox("univariate analysis",False):
                    if st.sidebar.checkbox("histogram plot",False):
                        continuos_selected = st.selectbox("Select a column",continous)
                        bin_value = st.slider('bin_value',0,100,10,1)
                        col=st.beta_color_picker("pick a color","#00FFAA")
                        sns.distplot(df[continuos_selected],kde=False,bins=bin_value,color=col)
                        sns.despine()
                        st.pyplot()
                    if st.sidebar.checkbox("density plot",False):
                        continuos_selected = st.selectbox("Select a column for density plot",continous)
                        col=st.beta_color_picker("Pick a color","#00FFAA")
                        sns.distplot(df[continuos_selected],hist=False,color=col)
                        sns.despine()
                        st.pyplot()
                    if st.sidebar.checkbox("box plot",False):
                        col=st.beta_color_picker("Pick a color","#00FFAA")
                        box_col = st.selectbox("select a column",continous)
                        sns.boxplot(y=box_col,data=df,color=col)
                        sns.despine()
                        st.pyplot()
                if st.sidebar.checkbox("bivariate plot",False):
                    if(st.sidebar.checkbox("Scatter plot",False)):
                        #select column 1
                        col1 =st.selectbox("Select col1 for scatter plot",continous)
                        col2 =st.selectbox("Select col2 for scatter plot",continous)
                        hue_component = st.selectbox("Select hue component",categorical)
                        if(col1 is not None and col2 is not None):
                            if(st.checkbox("View with a hue component",False)):
                                sns.relplot(x=col1,y=col2,data =df,hue=hue_component)
                                sns.despine()
                                st.pyplot() 
                            else:
                                sns.relplot(x=col1,y=col2,data =df)
                                sns.despine()
                                st.pyplot()
                    if(st.sidebar.checkbox("Box plot",False)):
                        y = st.selectbox("Select a column for box plot",continous)
                        x =st.selectbox("Select column",categorical)
                        col=st.beta_color_picker("Pick a color","#00FFAA")
                        sns.boxplot(x=x,y=y,data = df,color=col)
                        sns.despine()
                        st.pyplot()
                    

        if(len(categorical)==0):
            st.warning("There is no categorical data")
        else:
            if st.sidebar.checkbox("Explore categorical",False):
                ##plotting categorical plot 
                ##count plot 
                if st.sidebar.checkbox("Count plot",False):
                    column_count = st.selectbox("Select a column for count plot",categorical)
                    sns.countplot(x=column_count,data = df)
                    sns.despine()
                    st.pyplot()
                if st.sidebar.checkbox("Bar plot",False):
                    column_bar1 =st.selectbox("Select a column for bar plot",categorical)
                    column_bar2 = st.selectbox("Select column2 for bar plot",categorical)
                    sns.barplot(x=column_bar1,y=column_bar2,data =df)
                    sns.despine()
                    st.pyplot()
        if st.sidebar.checkbox("Plot heat map",False):
            temp_df = pd.get_dummies(df.corr())
            sns.heatmap(temp_df,annot=True,cmap="BuPu")
            st.pyplot()




if __name__=="__main__":
    main()
