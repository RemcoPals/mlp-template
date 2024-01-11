import streamlit as st
import joblib
from project_name.models.model import Model
from project_name.data.preprocesing import Preprocesing

model = joblib.load()


def predict_emoji(user_input):
    model.encode(user_input)
    output = model.predict(user_input)
    return output


def main():
    st.title('Emoji Prediction')
    inputt = st.text_area('Enter your tweet :)', '')

    if st.button('Predict Emoji'):
        # Make prediction when the button is clicked
        if inputt:
            predicted_emoji = predict_emoji(inputt)
            st.success(f'Predicted Emoji: {predicted_emoji}')
        else:
            st.warning('Please enter a tweet.')


if __name__ == "__main__":
    main()
