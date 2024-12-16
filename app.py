import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load GPT-Neo model and tokenizer locally from the specified path
model_path = "F:/huggingface_cache/models--EleutherAI--gpt-neo-1.3B/snapshots/dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
gpt_model = GPTNeoForCausalLM.from_pretrained(model_path)

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create the vector store (FAISS index)
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")  # Replace with a Hugging Face model
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Function to generate a response using GPT-Neo model
def get_gpt_response(context, user_question):
    # Updated prompt template with placeholders
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.

    Context: {context}
    Question: {user_question}

    Answer:
    """

    # Tokenize the input prompt
    input_ids = gpt_tokenizer(prompt, return_tensors='pt').input_ids

    # Check if the length exceeds max length (512 tokens for GPT-Neo 1.3B)
    max_length = 512  # Or set this as per your token limit
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]  # Truncate input to fit within model's limit

    # Generate the response using the GPT-Neo model
    output = gpt_model.generate(input_ids, max_new_tokens=100, num_return_sequences=1)  # Use max_new_tokens instead of max_length

    # Decode the generated text
    response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    return response



# Function to handle user input
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")  # Use Hugging Face embeddings

    # Load the FAISS index with the embeddings, allowing dangerous deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Search for similar documents
    docs = new_db.similarity_search(user_question, k=3)  # Search for similar documents

    # Combine the documents into a single context for the prompt
    context = "\n".join([doc.page_content for doc in docs])  # Access the 'page_content' attribute

    # Get response from the GPT-Neo model
    response = get_gpt_response(context, user_question)

    st.write("Reply:", response)

# Main function for the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using GPT-Neo Model 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
