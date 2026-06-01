from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("API_KEY")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOpenAI(model="gpt-3.5-turbo", 
                 temperature=0.9, 
                 openai_api_key=openai_api_key)



prompt_template = PromptTemplate(
    template="""You are a friendly and knowledgeable tea cultivation assistant. 
You help farmers with tea diseases, fertilizers, and best practices.

Previous conversation history (for context):
{chat_history}

Current question: {question}

Relevant information from tea documents:
{context}

Instructions:
1. Use the conversation history to understand follow-up questions
2. Answer based ONLY on the provided context and history
3. If you don't know, say "I don't have information about that in my tea knowledge base"
4. Be specific with fertilizer NPK ratios, disease symptoms, and control measures
5. Keep answers practical for farmers
6. If the user says "thank you" or "thanks", respond warmly and offer more help

Answer:""",
    input_variables=["chat_history", "question", "context"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)



    
conversational_chain = ConversationalRetrievalChain.from_llm(
       # llm=setup_local_llm(),  # this is for huggingface
        llm=llm,  # Using OpenAI's GPT-3.5
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
)
    

def ask_question(question):

    response = conversational_chain.invoke(
        {
            "question": question
        }
    )

    return response["answer"]

