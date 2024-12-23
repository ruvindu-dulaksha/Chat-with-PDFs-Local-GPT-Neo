# Chat with PDFs using GPT-Neo

This Streamlit application allows users to upload PDF files, process their contents into a vector store, and ask questions based on the text from the PDFs. The application leverages **GPT-Neo** for question answering and FAISS for document similarity search.

## Features

- **Upload and Process PDFs**: Extracts text from uploaded PDF files.
- **Text Splitting**: Breaks down the extracted text into manageable chunks.
- **Vector Store Creation**: Creates a FAISS index using embeddings for efficient similarity search.
- **GPT-Neo Integration**: Generates detailed answers to user questions based on the provided context from the PDFs.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `streamlit`
  - `PyPDF2`
  - `langchain`
  - `transformers`
  - `faiss`
  - `dotenv`
  - `huggingface_hub`

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ruvindu-dulaksha/Chat-with-PDFs-Local-GPT-Neo.git
cd <repository-folder>
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root and add the following:
```env
HUGGINGFACE_API_KEY=<your-huggingface-api-key>
```
Replace `<your-huggingface-api-key>` with your Hugging Face API key.

### 4. Configure GPT-Neo Model
Ensure the GPT-Neo model is downloaded and available at the specified path in the code (`model_path`). Update the `model_path` variable if needed:
```python
model_path = "<path-to-your-local-gpt-neo-model>"
```

### 5. Run the Application
Launch the Streamlit app:
```bash
streamlit run app.py
```

## How It Works

1. **Uploading PDFs**:
   - Upload one or multiple PDF files through the sidebar menu.
   - Click the **Submit & Process** button to extract and process the text.

2. **Processing Text**:
   - Text from the PDFs is extracted using `PyPDF2`.
   - The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
   - A FAISS vector store is created using Hugging Face embeddings.

3. **Asking Questions**:
   - Input your question in the main text input field.
   - The application searches the FAISS index for similar documents.
   - GPT-Neo generates a detailed answer based on the retrieved context.

## File Structure

```plaintext
project-root/
├── app.py               # Main Streamlit application code
├── requirements.txt     # List of required Python libraries
├── .env                 # Environment variables (not included in repo)
└── faiss_index/         # Directory for FAISS index storage
```

## Customization

- **Chunk Size and Overlap**:
  Modify `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` to adjust the text splitting behavior:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  ```

- **FAISS Embeddings**:
  Change the embeddings model by updating the `HuggingFaceEmbeddings` configuration:
  ```python
  embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
  ```

- **Response Generation**:
  Customize the GPT-Neo prompt template in the `get_gpt_response` function:
  ```python
  prompt = f"""
  Answer the question as detailed as possible from the provided context...
  """
  ```

## Known Issues

- Large PDFs might take longer to process. Consider splitting them into smaller files.
- Ensure the FAISS index folder is writable to save the vector store.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io) for the user interface.
- [Hugging Face](https://huggingface.co) for pre-trained embeddings and models.
- [PyPDF2](https://pypdf2.readthedocs.io/en/latest/) for PDF text extraction.
- [FAISS](https://faiss.ai) for efficient similarity search.
