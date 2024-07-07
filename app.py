import streamlit as st
from transformers import pipeline
import torch

# Load the QA pipeline
@st.cache
repo_id = 'sylvestr/roberta-finetuned-squad-v2'  # Replace with your Hugging Face repo ID
qa_pipeline = pipeline('question-answering', model=repo_id, tokenizer=repo_id, handle_impossible_answer=True)

# Streamlit app
st.title("Question Answering with BERT")

st.write("Enter a context and a question to get an answer from the fine-tuned BERT model.")

# Input context and question
context = st.text_area("Context", height=200)
question = st.text_input("Question")

# Get the answer
if st.button("Get Answer"):
    if context and question:
        input = {'question': question, 'context': context}
        response = qa_pipeline(**input)
        if response['answer'] == '':
            answer = "I'm sorry, the context you provided doesn't contain an answer to you question"
        else:
            answer = response['answer']
        st.write("**Answer:**", answer)
    else:
        st.write("Please provide both context and question.")
