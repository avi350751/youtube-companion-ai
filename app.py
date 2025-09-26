#The first things in your code must be#
__import__('sqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


#Import libraries
import streamlit as st
from helper import (
    get_video_id, get_transcript_text, translate_transcript, get_important_topics, generate_notes,
    create_chunks, create_vectorstore, rag_output)

# Main page---#
st.header("Your YouTube Video Companion")
st.write("Upload a YouTube video and interact with it in various ways.")

#sidebar inputs--#
with st.sidebar:
    st.title("Video Companion AI")
    st.markdown("------")
    st.markdown("Transform any youtube video into an interactive podcast or summarize the content")
    st.markdown("Input details")

    yt_url = st.text_input("YouTube URL")
    language = st.selectbox("Language",["en","hi"])

    task = st.radio("Choose what you want to do",
                    ["Chat with Video", "Notes for you"])
    
    submit_btn = st.button("Submit your choice --- Start Processing")
    st.markdown("--------------------")

    
    

if submit_btn:
    if yt_url and language:
        st.write("Processing started. Please wait...")
        vid = get_video_id(yt_url)
        if vid:
            with st.spinner("Step 1/3 : Fetching transcript...."):
                entire_transcript = get_transcript_text(vid, language)
                if language != "en":
                    with st.spinner("Step 1.5/3 : Translating transcript to English..."):
                        entire_transcript = translate_transcript(entire_transcript)
            if task == "Notes for you":
                with st.spinner("Step 2/3 : Extracting important topics..."):
                    imp_topics = get_important_topics(entire_transcript)
                    st.markdown("### Here are the important topics discussed in the video")
                    st.write(imp_topics)
                with st.spinner("Step 3/3 : Generating notes..."):
                    notes = generate_notes(entire_transcript)
                    st.markdown("### Here are the notes for you")
                    st.write(notes)

                st.success("Process completed!")

            if task == "Chat with Video":
                with st.spinner("Step 2/3 : Creating chunks and vector store..."):
                    chunks = create_chunks(entire_transcript)
                    vectorstore = create_vectorstore(chunks)
                    st.session_state['vector_store'] = vectorstore
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                st.success("Step 2/3 completed! You can now chat with the video.")
                
                
#Chatbot Session
if task == "Chat with Video" and 'vector_store' in st.session_state:
    st.divider()
    st.markdown("### Chat with the video")

    #Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    #user input
    user_query = st.chat_input("Enter your query about the video:")
    if user_query:
        st.session_state.messages.append({"role":"user","content":user_query})
        with st.spinner("Generating response..."):
            with st.chat_message("user"):
                st.write(f"Your query: {user_query}")
            
            with st.chat_message("assistant"):
                response = rag_output(user_query, st.session_state['vector_store'])
                st.markdown("#### Response from the video")
                st.write(response)
            st.session_state.messages.append({"role":"assistant","content":response})
            #st.success("Response generated!")
