import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PromptTemplate import PromptTemplates
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

class ChatbotLogic:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def query(self, input_text, chat_history):

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptTemplates.CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        prompt_templates = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt_templates)
        rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

        formatted_history = []
        for exchange in chat_history:
            if len(exchange) > 0:
                formatted_history.append(HumanMessage(content=exchange[0]))
            if len(exchange) > 1:
                formatted_history.append(AIMessage(content=exchange[1]))

        try:
            response = rag_chain.invoke({
                "input": input_text,
                "chat_history": formatted_history
            })
            print("response: ", response)
            return response['answer']
        except Exception as e:
            print("Error during query: ", e)
            return {"answer": "Something went wrong while processing your message."}

# if __name__ == "__main__":

#     pdf_path = r"C:\Users\Kulde\OneDrive\Desktop\LLM Projects\BoT\gpt\app\priyank satani resume.pdf"
#     documents = PyPDFLoader(pdf_path).load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     splited_docs = splitter.split_documents(documents)

#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)
#     embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base")

#     vectorstore = Chroma.from_documents(
#         documents=splited_docs,
#         embedding=embedding_model,
#         persist_directory='priyank'
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
#     chatbot = ChatbotLogic(llm, retriever)

#     chat_history = []

#     print("Welcome to the Conversational Q&A Bot (Backend Test)!")
#     while True:
#         input_text = input("User: ")
#         if input_text.lower() in ["quit",'q', "exit", "stop"]:
#             break
#         response = chatbot.query(input_text, chat_history)
#         print("Assistant:", response)
#         chat_history.extend([input_text, response])
#     print("Exiting.")