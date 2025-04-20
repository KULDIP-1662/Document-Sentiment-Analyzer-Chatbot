import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

async def load_pdf(file_object: UploadFile, filename: str, embedding_model):
    import os
    persist_directory = f"Chroma_db/{filename}"  # Or your desired directory

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        
        print("🔁 Loading vectorstore from disk...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model)

        return vectorstore.as_retriever()
    
    else:
        # Create the directory if it doesn't exist
        print("⚙️ Creating vectorstore and saving to disk...")
        try:
            contents = await file_object.read()

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(contents)
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=persist_directory)
            retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
            return retriever

        finally:
            import os
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# import os
# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# def load_pdf(file_object: UploadFile, filename: str, embedding_model):

#     if file_object is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(file_object.read())
#             tmp_path = tmp_file.name

#         documents = PyPDFLoader(tmp_path).load()
#         os.remove(tmp_path)

#         splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         )

#         splited_docs = splitter.split_documents(documents)

#         persist_directory = f'Chroma_DB/{filename}'

#         if os.path.exists(persist_directory) and os.listdir(persist_directory):
#             print("🔁 Loading vectorstore from disk...")
#             vectorstore = Chroma(
#                 persist_directory=persist_directory,
#                 embedding_function=embedding_model
#             )

#         else:
#             print("⚙️ Creating vectorstore and saving to disk...")

#             vectorstore = Chroma.from_documents(
#                 documents=splited_docs,
#                 embedding=embedding_model,
#                 persist_directory=persist_directory
#             )

#         retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
#         return retriever














# import os
# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# def load_pdf(file_object, file_name, embedding_model):
    
#     if file_object is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(file_object.read())
#             tmp_path = tmp_file.name

#         documents = PyPDFLoader(tmp_path).load()
#         print(documents)
#         os.remove(tmp_path)

#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         splited_docs = splitter.split_documents(documents)

#         if os.path.exists(f'./ChormaDB/{file_name}'):
#             vectorstore = Chroma(persist_directory = f'./ChormaDB/{file_name}')
#             print("Loading existing ChromaDB...")
#         else:
#             vectorstore = Chroma.from_documents(
#                 documents=splited_docs,
#                 embedding=embedding_model,
#                 persist_directory = f'./ChormaDB/{file_name}'
#             )
#             print("Creating new ChromaDB...")

#         retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
#         return retriever
