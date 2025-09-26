import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


#Function to extract video id from youtoube url
def get_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

# Function to get the transcript of this video
def get_transcript_text(video_id, language):
    yt_api = YouTubeTranscriptApi()
    try:
        transcript = yt_api.fetch(video_id, languages=[language])
        entire_transcript =" ".join([i.text for i in transcript])
        time.sleep(10)
        return entire_transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None
    

# Function to translate the transcript into English
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
            You are an expert translator with deep cultural and ligusitic understanding.
            I will provide you with a transcript. Your task is to translate it into English with absolute accuracy keeping the
            original meaning, tone, and context intact. Ensure there is no addition and no ommision.
            - Tone and Style: Formal/Informal , Emotional/Natural as in original
            - Nuances, Idioms, Cultutal references : Adapt appropriately while keeping the original intent
            - Speakers' voice : Miantains ame perspective, no rewriting into third-person
            Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possile to the original intent.
    
        Transcript: {transcript}""")

        #runnable
        chain = prompt|llm

        #run chain
        response = chain.invoke({"transcript":transcript})

        return response.content
    except Exception as e:
        st.error(f"Error translating transcript: {e}")
        return None


#Function to get most important topics
def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant that extracts 5 most important topic from the video transcript or summary in bulleted points.
            Rules:
                 - Summarize into exactly 5 major points
                 - Each point need to be a key topic, not a subtopic or smaller details.
                - Keep wording concise , clear and to the point.
                - Do not phrase topic as a question or opinions.
                - Avoid redundancy, show topic which is only described in the transcript
                                                  
            Here is the transcript: {transcript}""")

        #runnable
        chain = prompt|llm

        #run chain
        response = chain.invoke({"transcript":transcript})
        return response.content
    except Exception as e:
        st.error(f"Error extracting topics: {e}")
        return None
    
#Function to generate notes from the transcript
def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
            You are an expert AI-note taker. Your task is to go through the following Youtube video transcript
            and generate well-structured, lucid and concise notes.
            Rules:
                 - Use bulleted points , grouped into clear sections with headings.
                - Highlight key takeaways, important details, and any action items.
                - Keep wording concise , clear and to the point.
                - If the transcript contains multiple themes or topics, organize them with subheadings.
                - Avoid redundancy and do no add any information not present in the transcript.
                                                  
            Here is the transcript: {transcript}""")
        
        #runnable
        chain = prompt|llm

        #chain
        response = chain.invoke({"transcript":transcript})
        return response.content
    except Exception as e:
        st.error(f"Error generating notes: {e}")
        return None


#Function to create chunks of text
def create_chunks(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    doc = text_splitter.create_documents([transcript])
    return doc

#Function to create embeddings and store in vector db
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",transport="grpc")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

#RAG Function to chat with video
def rag_output(query, vectorstore):
    try:
        relevant_docs = vectorstore.similarity_search(query, k=4)
        context = " ".join([doc.page_content for doc in relevant_docs])

        prompt = ChatPromptTemplate.from_template("""
            You are an kind, polite and expert assistant. 
            Begin with a warm greeting (avoid repeating the greeting eevry time).
            Understand the user's intent even with typos or grammatical errors.
            If you don't know the answer, or not in context just say " I couldn't find the answer in the database. Could you rephrase or ask something else"
            Keep the answers clear, concise and to the point.
            Context: {context}
            User Question: {question}
            Answer : 
            """)

        #runnable
        chain = prompt|llm

        #run chain
        response = chain.invoke({"context":context, "question":query})
        return response.content
    except Exception as e:
        st.error(f"Error in RAG output: {e}")
        return None

