import streamlit as st
import numpy as np
import joblib

st.title("ELECTION VOTECOUNT PREDICTION")
st.markdown(
    """
    <style>
    body {
        background-image: url("https://cloudfront.penguin.co.in/wp-content/uploads/2019/05/Election_Header.jpg");
        background-size: cover;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color:#10B10D;
        text-align: center;
        font-family: Unispace,Unispace;
    }
    .stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton>button {
        background-color:#0D06B5;
        color:#FFFFFF;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }.prediction-box {
        background-color: #FFFFFF;
        border: 2px solid #104A7C;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        color:#104A7C;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the saved model and encoders
model = joblib.load('main_project/model.pkl')
lb_st_name = joblib.load('main_project/lb_st_name.pkl')
lb_pc_type = joblib.load('main_project/lb_pc_type.pkl')
lb_cand_sex = joblib.load('main_project/lb_cand_sex.pkl')
lb_partyname = joblib.load('main_project/lb_partyname.pkl')
scaler = joblib.load('main_project/st_scalar.pkl')


# Create input fields for the user to enter the required data
state_choice = st.selectbox('Choose your state',lb_st_name.classes_)
partyname_choice = st.selectbox('Choose the party name',lb_partyname.classes_)
sex_choice = st.selectbox("Select the gender of candidate", lb_cand_sex.classes_)
category_choice = st.selectbox('Choose the category',lb_pc_type.classes_)
num_electors = st.number_input("Enter the number of electors:", min_value=1, max_value=10000000, step=1, value=100)

if st.button('Predict'):
    if state_choice not in lb_st_name.classes_ or category_choice not in lb_pc_type.classes_ or partyname_choice not in lb_partyname.classes_:
        st.error("One of the input values is not recognized.")
    else:
        state_encoded = lb_st_name.transform([state_choice])[0]
        pc_type_encoded = lb_pc_type.transform([category_choice])[0]
        sex_encoded = lb_cand_sex.transform([sex_choice])[0]
        party_encoded = lb_partyname.transform([partyname_choice])[0]

        # Prepare the input data for the model
        input_data = np.array([[state_encoded, pc_type_encoded, party_encoded, num_electors]])

        # Make predictions
        if input_data.shape[1] == 4:
            prediction = model.predict(input_data)
            st.markdown(
                f"""
                <div class="prediction-box">
                    TOTAL VOTE COUNT: {prediction[0]}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(f"Expected 4 features, but got {input_data.shape[1]} features")