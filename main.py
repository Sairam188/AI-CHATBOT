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
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", torch_dtype=torch.float16, device=0)


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
        # Generate response using the LLM
        response = chatbot(user_input, max_length=300, num_return_sequences=1)
        return response[0]['generated_text']


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
