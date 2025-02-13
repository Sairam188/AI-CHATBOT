import streamlit as st
from transformers import pipeline
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a more advanced model (Mistral-7B-Instruct or Llama-2-7B)
chatbot = pipeline("question-answering", model="deepset/roberta-base-squad2")




def healthcare_chatbot(user_input):
    """Generates chatbot response using AI with additional rule-based logic."""
    # Simple rule-based handling
    if "symptom" in user_input.lower():
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input.lower():
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input.lower():
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        user_input = user_input.lower()

         predefined_responses = {
        "fever": "A fever is a temporary increase in body temperature, often due to an infection. Stay hydrated and rest. If it persists, consult a doctor.",
        "corona precautions": "To prevent COVID-19, wear a mask, maintain social distancing, wash hands frequently, and get vaccinated.",
        "symptoms of covid": "Common symptoms include fever, cough, fatigue, difficulty breathing, and loss of taste or smell.",
        "medication": "Always take medications as prescribed by your doctor. If you experience side effects, seek medical advice."
        }

        # Check if input matches predefined responses
        for key in predefined_responses:
            if key in user_input:
               return predefined_responses[key]

        # Use AI model for other queries
        context = """This is a healthcare chatbot that provides medical guidance. 
                 It can answer general health-related questions but is not a substitute for professional medical advice."""
    
        response = chatbot(question=user_input, context=context)  # Fix: Pass question & context correctly
        return response['answer']

def main():
    """Streamlit Web App UI for Chatbot."""
    st.set_page_config(page_title="Healthcare Chatbot", page_icon="💬")
    st.title("🩺 Healthcare Assistant Chatbot")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        st.markdown(msg)

    # User Input
    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input:
            response = healthcare_chatbot(user_input)
            st.session_state["messages"].append(f"**User:** {user_input}")
            st.session_state["messages"].append(f"**Healthcare Assistant:** {response}")
            st.rerun()


if __name__ == "__main__":
    main()
