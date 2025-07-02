'''import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,LabelEncoder 
import tensorflow as tf 
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st

model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gende.pkl','rb') as file:
    label_encoder_gende=pickle.load(file)
    
with open('One_Hot_Encoder','rb') as file:
    One_Hot_Encoder=pickle.load(file)
  




# Save the encoder
#with open('One_Hot_Encoder', 'wb') as file:
 #   pickle.dump(encoder, file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
 # Streamlit APP
st.title("Customer Churn Prediction") 

# User Input
geography = st.selectbox('Geography', One_Hot_Encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gende.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])  

#Preparing Input Data
input_data = pd.DataFrame({'CreditScore': [credit_score],'Gender':[label_encoder_gende.transform([gender])[0]],
                          'Age': [age],'Tenure': [tenure],'Balance': [balance],'NumOfProducts': [num_of_products],
                          'HasCrCard': [has_cr_card],
                          'IsActiveMember': [is_active_member],
                          'EstimatedSalary': [estimated_salary]

})

# One-hot encode 'Geography'
geo_encoded = One_Hot_Encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=One_Hot_Encoder.get_feature_names_out(['Geography']))

# Combining One Hot Encoded with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling Input Data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The Customer will churn")
else :
    st.write("This Customer is not likely to churn")  
'''
import pandas as pd 
import numpy as np
import pickle
import tensorflow as tf 
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# Sample data
df = pd.DataFrame({'Geography': ['France', 'Germany', 'Spain']})

# Create and fit encoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df[['Geography']])

# Save the encoder properly
with open('One_Hot_Encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("✅ Encoder saved as One_Hot_Encoder.pkl")


# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load label encoder for gender
with open('label_encoder_gende.pkl', 'rb') as file:
    label_encoder_gende = pickle.load(file)

# Load one-hot encoder for geography (newly generated and saved separately)
with open('One_Hot_Encoder.pkl', 'rb') as file:
    One_Hot_Encoder = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App UI
st.title("Customer Churn Prediction")

# User Input Section
geography = st.selectbox('Geography', One_Hot_Encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gende.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gende.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the 'Geography' column
geo_encoded = One_Hot_Encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=One_Hot_Encoder.get_feature_names_out(['Geography'])
)

# Combine encoded geography with the rest of the data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input features
input_data_scaled = scaler.transform(input_data)

# Make the prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"📊 **Churn Probability:** {prediction_proba * 100:.2f}%")


# Output the result
if prediction_proba > 0.5:
    st.markdown("🔴 **Prediction:** The customer is **likely to churn.**")
else:
    st.markdown("🟢 **Prediction:** The customer is **not likely to churn.**")
