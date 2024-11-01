import streamlit as st
import pandas as pd
import pickle

# Load the trained model with error handling
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except EOFError:
    st.error("The model file is empty or corrupted.")
    st.stop()
except FileNotFoundError:
    st.error("The model file was not found.")
    st.stop()

# Create a Streamlit app
st.title('Titanic Survival Prediction App')

def user_input_features():
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=80)
    fare = st.number_input('Fare', min_value=0, max_value=500)
    sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=8)
    parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=8)
    embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
    data = {'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Embarked': [embarked]}
    features = pd.DataFrame(data, index=[0])
    return features

# Encode categorical variables
def encode_features(df):
    df_encoded = df.copy()
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
    df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    return df_encoded

df = user_input_features()
encoded_df = encode_features(df)

# Ensure columns match the training data
try:
    feature_names = model.feature_names_in_
except AttributeError:
    # If the model does not have feature_names_in_, use hardcoded names
    feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']

# Ensure columns are in the right order
for col in feature_names:
    if col not in encoded_df.columns:
        encoded_df[col] = 0

# Reorder columns to match the training set
encoded_df = encoded_df[feature_names]

# Predict
if st.button('Predict'):
    try:
        prediction = model.predict(encoded_df)
        prediction_proba = model.predict_proba(encoded_df)
        st.write("Prediction Probability:", prediction_proba)
        if prediction[0] == 1:
            st.write('The passenger is predicted to survive.')
        else:
            st.write('The passenger is predicted not to survive.')
    except ValueError as e:
        st.error(f"Prediction error: {e}")
