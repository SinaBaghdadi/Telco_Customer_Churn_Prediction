import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns  
import scipy.stats as sta
from PIL import Image
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix,\
                            classification_report, auc, roc_curve
from sklearn.ensemble import AdaBoostClassifier,\
                             RandomForestClassifier,\
                              GradientBoostingClassifier
import streamlit as st
import plotly 
import pickle

warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
sns.set_palette("viridis", 3)


image = Image.open('data.JPG')
st.image(image, width=700,)

image1= Image.open('con_mat.PNG')  
image2= Image.open('ROC_Curve.PNG')

Grboost_model = pickle.load(open('Grboost_model.sav', 'rb'))

strings = {
    "gender": ['Female', 'Male'],
    "Partner": ['Yes', 'No'],
    "SeniorCitizen": ['No', 'Yes'],
    "Dependents": ['No', 'Yes'],
    "PhoneService": ['No', 'Yes'],
    "PaperlessBilling": ['Yes', 'No'],
    "MultipleLines": ['No phone service', 'No', 'Yes'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
    "OnlineSecurity": ['No', 'Yes', 'No internet service'],
    "OnlineBackup": ['Yes', 'No', 'No internet service'],
    "DeviceProtection": ['No', 'Yes', 'No internet service'],
    "TechSupport": ['No', 'Yes', 'No internet service'],
    "StreamingTV": ['No', 'Yes', 'No internet service'],
    "StreamingMovies": ['No', 'Yes', 'No internet service'],
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "PaymentMethod": ['Electronic check',
                      'Mailed check',
                      'Bank transfer (automatic)',
                      'Credit card (automatic)']
}

# min, max, default value
ints = {
    "tenure": [0, 100, 2],
    "MonthlyCharges": [0, 1000, 100],
    "TotalCharges": [0, 50000, 1000]
}

labels = ["No Churn", "Churn"]


def predicter(df): 
    df.replace('Yes', 1, inplace=True)
    df.replace('No', 0, inplace=True)
    df.replace('No internet service', 0, inplace=True)
    df.replace('No phone service', 0, inplace=True)
    df.replace('Fiber optic', 2, inplace=True)
    df.replace('DSL',1, inplace=True)
    df.replace('Male',1, inplace=True)
    df.replace('Female',0, inplace=True)
    df = pd.get_dummies(data=df, columns=['Contract', 'PaymentMethod'])
    for i in strings['Contract'] :
        if 'Contract_'+ i not in df.columns.to_list():
            df['Contract_'+ i] = 0
    for i in strings['PaymentMethod'] :
        if 'PaymentMethod_'+ i not in df.columns.to_list():
            df['PaymentMethod_'+ i] = 0
    df.sort_index(axis=1, inplace=True)
    y_pred = Grboost_model.predict(df)
    y_prob = Grboost_model.predict_proba(df)
    return(y_pred, y_prob )

 

st.title('Telecom Customer Churn Prediction')
#st.header('')
st.subheader('Input the following information to make prediction')


str_form = {}
input_form = {}
col1, col2, col3, col4 = st.columns(4)
col5, col6 = st.columns(2)



with st.sidebar:
     st.header('Will a customer stay or leave?')
     st.write('In recent years, the telecom market has been very competitive. The cost of retaining existing\
               telecom customers is lower than attracting new customers. It is necessary for a telecom company to\
               understand customer churn through customer relationship management (CRM). Therefore, CRM\
               analyzers are required to predict which customers will churn. This App proposes a customer-churn\
               prediction system that uses an ensemble-learning technique consisting of machine-learning algorithms are\
               selected to build a model.')
     st.subheader('Confusion matrix :') 
     st.image(image1) 
     st.subheader('Receiver operating characteristic curve:')
     st.image(image2)    
with st.form('input_form', ):
    with col1:
        tenure = st.number_input('tenure', min_value=0, max_value=100)
        MonthlyCharges = st.number_input('Monthly Charges', min_value=0, max_value=1000)
        TotalCharges = st.number_input('Total Charges', min_value=0, max_value=50000)
        num_form = {'tenure':tenure,
                    'MonthlyCharges': MonthlyCharges,
                    'TotalCharges': TotalCharges
                   }

    with col2 :
        for i,j in strings.items():
            if len(j)>2 and len(i)<=12 :
                z = st.selectbox(i, j)
                str_form.update({i:z})

    with col3 :
        for i,j in strings.items():
            if len(j)>2 and len(i)>12 :
                z = st.selectbox(i, j)
                str_form.update({i:z})


    with col4 :
        for i,j in strings.items():
            if len(j)<3 :
                z = st.radio(i, j)
                str_form.update({i:z})

    
    submit = st.form_submit_button('Submit')
            
if submit :
   str_form.update(num_form)
   input_form = str_form
   input_df = pd.DataFrame([input_form])
   st.write(input_df)
   pred = predicter(input_df)
   progress_bar = st.progress(0)

   with col5:
        st.write('Churn Class:', pred[0][0])
        st.write('Churn prob:', pred[1])

   with col6:

        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(pred[1][0], autopct='%.1f%%', labels=labels)
        st.pyplot(fig)

   progress_bar.progress(100)

