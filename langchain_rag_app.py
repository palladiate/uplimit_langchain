from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from operator import itemgetter

load_dotenv('.env.dev')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load PDF file
loader = PyMuPDFLoader('levothyroxine.pdf')
data = loader.load()

# Split and Embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.from_documents(documents=splits, embedding=embedding_model)

# Create Prompt
qna_prompt = """
    You are a trained pharmacist who is an expert in reading medical documentation
    and will answer any questions related to prescription medication you know. You 
    like to explain things simply for laymen and non-medical questioners. You are 
    comfortable to say you do not know if you cannot find the answer. You do your 
    best to find the exact references that lead you to believe the answer is true.
    
    You will be given a question and a context to guide your answer.
    
    Question: {question}
    Context: {context}
    Answers
"""
qna_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful AI pharmacist"),
            HumanMessagePromptTemplate.from_template(qna_prompt),
        ]
)

# Create LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# Create Retriever that adds the context
retriever = vector_store.as_retriever()

# Do the Post-processing on the output of the retriever
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG Chain
rag_chain = (
    {"context": itemgetter("input") | retriever | format_docs,
     "question": RunnablePassthrough()}
    | qna_prompt_template
    | llm
    | StrOutputParser()
)

def provide_bot_response(user_question):
    rag_response = rag_chain.invoke({"input": user_question})
    return rag_response

def main():
    rag_response = rag_chain.invoke({"input": "Can I give this medicine to my child?"})
    print (rag_response)

if __name__ == "__main__":
    main()
