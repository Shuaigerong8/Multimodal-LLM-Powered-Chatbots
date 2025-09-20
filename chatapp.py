from concurrent.futures import ThreadPoolExecutor
import os
import time
import json
import glob
import re
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from pytesseract import pytesseract
from gtts import gTTS
import threading
import base64


# Langchain and Ollama imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pandas as pd
from ollama import generate

# Cohere import
import cohere
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import io

# Configure Tesseract executable path
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Google Generative AI API
genai.configure(api_key=st.secrets["google"]["api_key"])  # Use Streamlit secrets

# Configure Cohere API
co = cohere.Client(st.secrets["cohere"]["api_key"])

# Initialize Llama model
llama_model = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")  # Ensure base_url matches Ollama's

# Load or create a DataFrame to store image descriptions
def load_or_create_dataframe(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=['image_file', 'description'])

df = load_or_create_dataframe('image_descriptions.csv')

# Function to set page styles (from app.py)
def set_page_style():
    st.markdown("""
    <style>
        /* Page background */
        body {
            background-color: #4e604c; /* Dark green background */
            color: #FFFFFF;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Unified font */
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
        }
        /* Sidebar style */
        .css-1d391kg {
            background-color: #779377; /* Light green */
            padding: 20px;
            width: 250px;  /* Fixed sidebar width */
        }
        .css-1aumxhk {
            color: #FFFFFF;
        }
        .main-content {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto; /* Add scrollbar */
        }
        /* Title styles */
        h1, h2, h3, h4, h5, h6 {
            color: #a5b286;
            margin-top: 0;
        }
        /* Button styles */
        .stButton > button {
            background-color: #73776b;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            cursor: pointer; /* Add cursor style */
        }
        .stButton > button:hover {
            background-color: #5c5a4d;
        }
        .stTextInput > div > div > input {
            background-color: #f4f4f4; /* Chat input box background color */
            color: #264653; /* Input text color */
            border: 1px solid #73776b;
            border-radius: 5px;
            padding: 10px;
        }
        .user-message, .ai-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            color: #FFFFFF;
        }
        /* User message bubbles */
        .user-message {
            background-color: #779377;
            text-align: right;
        }
        /* AI reply bubbles */
        .ai-message {
            background-color: #a5b286;
            text-align: left;
            color: #264653;
        }
        /* Loading style */
        .loading {
            text-align: center;
            color: #73776b;
            font-weight: bold;
        }
        /* Scrollbar style */
        ::-webkit-scrollbar {
            width: 12px;
        }
        ::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #73776b;
            border-radius: 20px;
            border: 3px solid #f0f0f0;
        }
        /* Chat container styles */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }
        .chat-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 5px 0;
        }
        .chat-row.reverse {
            flex-direction: row-reverse;
        }
        .user-message-cohere, .ai-message-cohere {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
        }
        .user-message-cohere {
            background-color: #4caf50;
            color: white;
            text-align: left;
        }
        .ai-message-cohere {
            background-color: #f1f1f1;
            color: black;
            text-align: left;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #4caf50;
            color: white;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== Gemini Model Interface ====================

# Initialize directories
os.makedirs('conversations', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# ================================
# Text-to-Speech (TTS) Functions
# ================================

# Global variable to manage TTS playback
tts_playback_active = False

def play_text(text):
    global tts_playback_active
    tts_playback_active = True
    tts = gTTS(text, lang='en')
    tts.save("temp_audio.mp3")
    with open("temp_audio.mp3", "rb") as audio_file:
        audio_data = audio_file.read()
        encoded_audio = base64.b64encode(audio_data).decode()
    audio_html = f"""
    <audio autoplay id="tts-audio" onended="tts_playback_active = false;">
        <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def stop_text():
    global tts_playback_active
    tts_playback_active = False
    stop_html = """
    <script>
    const audio = document.getElementById("tts-audio");
    if (audio) {
        audio.pause();
        audio.currentTime = 0;
    }
    </script>
    """
    st.markdown(stop_html, unsafe_allow_html=True)

# ================================
# Text Extraction Functions
# ================================

def extract_text_from_pdf_gemini(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        return f"Error extracting PDF text: {e}"

def extract_text_from_txt_gemini(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        return f"Error extracting TXT text: {e}"

def extract_text_from_image_gemini(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        if text.strip():
            return text
        else:
            return "No text detected in the image."
    except Exception as e:
        return f"Error processing image: {e}"

def extract_text_from_audio_gemini(file):
    temp_file = "temp_audio_gemini.wav"
    try:
        audio = AudioSegment.from_file(file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        duration = len(audio)
        segment_length = 30000  # 30 seconds in milliseconds
        text_output = []

        recognizer = sr.Recognizer()

        for i in range(0, duration, segment_length):
            segment = audio[i:i + segment_length]
            segment.export(temp_file, format="wav")

            with sr.AudioFile(temp_file) as source:
                audio_data = recognizer.record(source)
                try:
                    segment_text = recognizer.recognize_google(audio_data, language="en-US")
                    text_output.append(segment_text)
                except sr.UnknownValueError:
                    text_output.append("[Unrecognized Segment]")
                except sr.RequestError as e:
                    text_output.append(f"[Error: {e}]")

        return " ".join(text_output)
    except Exception as e:
        return f"Error processing audio: {e}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ================================
# Utility Functions
# ================================

def strip_html_tags(text):
    """
    Removes all HTML tags from the given text.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def chat_with_gemini(user_message, chat_history, document_context=""):
    """
    Sends the entire chat history along with the document context to the Gemini model and retrieves the response.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # 构建完整的对话历史
        conversation = ""
        for message in chat_history:
            role = "You" if message["role"] == "user" else "AI"
            conversation += f"{role}: {message['content']}\n"
        
        # 添加当前用户消息
        conversation += f"You: {user_message}\nAI:"
        
        # 最终的提示
        prompt = f"Document context: {document_context}\n{conversation}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error interacting with the AI: {str(e)}"

def save_gemini_conversation():
    """
    Saves the current Gemini conversation to a JSON file.
    """
    if st.session_state.gemini_chat_history:
        conversation_id = st.session_state.gemini_current_conversation
        conversation_data = {
            "conversation_id": conversation_id,
            "chat_history": st.session_state.gemini_chat_history,
        }
        file_path = os.path.join('conversations', f"{conversation_id}.json")

        # 如果文件已经存在，覆盖它
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        st.session_state.gemini_conversations[conversation_id] = st.session_state.gemini_chat_history.copy()
        st.success(f"Conversation '{conversation_id}' saved successfully!")

def load_gemini_conversations():
    """
    Loads all saved Gemini conversations from the 'conversations' directory.
    """
    conversations = {}
    for file_name in glob.glob('conversations/*.json'):
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        conversation_id = data["conversation_id"]
        chat_history = data["chat_history"]
        conversations[conversation_id] = chat_history
    return conversations

# ================================
# Streamlit Interface
# ================================

def run_gemini_interface():
    st.title("Interactive Document Chat App - Gemini Model")

    # Load saved conversations into session_state
    if "gemini_conversations" not in st.session_state:
        st.session_state.gemini_conversations = load_gemini_conversations()

    # Initialize session state for Gemini
    if "gemini_chat_history" not in st.session_state:
        st.session_state.gemini_chat_history = []
    if "gemini_document_context" not in st.session_state:
        st.session_state.gemini_document_context = ""
    if "gemini_pending_response" not in st.session_state:
        st.session_state.gemini_pending_response = False
    if "gemini_current_conversation" not in st.session_state:
        st.session_state.gemini_current_conversation = f"Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    st.sidebar.title("Gemini Conversations")

    # New conversation button
    if st.sidebar.button("New Gemini Conversation"):
        st.session_state.gemini_chat_history = []
        st.session_state.gemini_document_context = ""
        st.session_state.gemini_current_conversation = f"Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        st.success("Started a new Gemini conversation.")

    # Select saved conversation
    saved_conversations = list(st.session_state.gemini_conversations.keys())
    selected_conversation = st.sidebar.selectbox("Select a saved conversation:", ["None"] + saved_conversations, key="gemini_selectbox")

    # Load selected conversation
    if selected_conversation != "None":
        st.session_state.gemini_chat_history = st.session_state.gemini_conversations[selected_conversation]
        st.session_state.gemini_current_conversation = selected_conversation

        # Delete conversation button
        if st.sidebar.button("Delete Selected Gemini Conversation"):
            file_path = os.path.join('conversations', f"{selected_conversation}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                st.success(f"Conversation '{selected_conversation}' has been deleted.")

            # Remove from session_state
            del st.session_state.gemini_conversations[selected_conversation]

            # Reset current conversation
            st.session_state.gemini_chat_history = []
            st.session_state.gemini_current_conversation = f"Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Save button
    if st.sidebar.button("Save Current Gemini Conversation"):
        save_gemini_conversation()

    # Sidebar for file upload
    st.sidebar.title("Gemini Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload files", 
        type=["pdf", "txt", "jpg", "png", "mp3", "wav"], 
        accept_multiple_files=True, 
        key="gemini_file_uploader"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            with st.spinner(f'Processing {uploaded_file.name}...'):
                if file_type == "application/pdf":
                    text = extract_text_from_pdf_gemini(uploaded_file)
                    st.session_state.gemini_document_context += f"\n{text}"
                elif file_type == "text/plain":
                    text = extract_text_from_txt_gemini(uploaded_file)
                    st.session_state.gemini_document_context += f"\n{text}"
                elif file_type in ["image/jpeg", "image/png"]:
                    text = extract_text_from_image_gemini(uploaded_file)
                    st.session_state.gemini_document_context += f"\n{text}"
                elif file_type in ["audio/mpeg", "audio/wav"]:
                    text = extract_text_from_audio_gemini(uploaded_file)
                    st.session_state.gemini_document_context += f"\n{text}"
                else:
                    text = "Unsupported file type."
                    st.session_state.gemini_document_context += f"\n{text}"
                st.success(f'Finished processing {uploaded_file.name}.')

    # Chat display with TTS buttons
    chat_placeholder = st.empty()

    def display_chat_history():
        with chat_placeholder.container():
            st.write("### Chat History")
            for index, message in enumerate(st.session_state.gemini_chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns([8, 2])
                    with col1:
                        content = strip_html_tags(message['content'])
                        st.markdown(f"""
                        <div class="ai-message">
                            <strong>AI:</strong> {content}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        play_key = f"tts_play_{st.session_state.gemini_current_conversation}_{index}"
                        stop_key = f"tts_stop_{st.session_state.gemini_current_conversation}_{index}"
                        
                        if st.button("🔊 Play", key=play_key):
                            play_text(message["content"])
                        if st.button("⏹ Stop", key=stop_key):
                            stop_text()

    display_chat_history()

    # Chat input form
    with st.form(key="gemini_chat_form"):
        user_input = st.text_input("Gemini Chat Input", placeholder="Type your message here...", key="gemini_user_input")
        submit_button = st.form_submit_button("Send")

        if submit_button:
            if user_input.strip():
                if st.session_state.gemini_pending_response:
                    st.warning("Please wait for the current response to finish.")
                else:
                    # 限制对话历史长度
                    MAX_HISTORY_LENGTH = 10  # 根据需要调整
                    if len(st.session_state.gemini_chat_history) >= MAX_HISTORY_LENGTH:
                        st.session_state.gemini_chat_history = st.session_state.gemini_chat_history[-(MAX_HISTORY_LENGTH-1):]
                    
                    st.session_state.gemini_chat_history.append({"role": "user", "content": user_input})
                    st.session_state.gemini_pending_response = True

    # Process pending response
    if st.session_state.gemini_pending_response:
        if len(st.session_state.gemini_chat_history) > 0:
            last_message = st.session_state.gemini_chat_history[-1]
            if last_message["role"] == "user":
                with chat_placeholder.container():
                    st.markdown("<div class='loading'>Generating response...</div>", unsafe_allow_html=True)
                    # Consider using asyncio or threading for non-blocking operations
                    time.sleep(2)  # Simulate response delay

                # Generate AI response, passing the full chat history
                ai_response = chat_with_gemini(
                    last_message["content"], 
                    st.session_state.gemini_chat_history, 
                    st.session_state.gemini_document_context
                )
                ai_response = strip_html_tags(ai_response)

                st.session_state.gemini_chat_history.append({"role": "ai", "content": ai_response})
                st.session_state.gemini_pending_response = False

                # Save to history as AI responds
                save_gemini_conversation()

                # Refresh chat history
                chat_placeholder.empty()
                with chat_placeholder.container():
                    for index, message in enumerate(st.session_state.gemini_chat_history):
                        if message["role"] == "user":
                            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                        else:
                            col1, col2 = st.columns([8, 2])
                            with col1:
                                st.markdown(f"<div class='ai-message'><strong>AI:</strong> {message['content']}</div>", unsafe_allow_html=True)
                            with col2:
                                if st.button("🔊 Play", key=f"tts_play_{index}"):
                                    play_text(message["content"])
                                if st.button("⏹ Stop", key=f"tts_stop_{index}"):
                                    stop_text()
                                    
# ==================== Llama Model Interface ====================

# Functions for Llama Model
def extract_text_from_pdf_llama(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if not text.strip():
            raise ValueError("No extractable text found in the uploaded PDF.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def describe_image_with_llava_llama(image_file):
    try:
        img = Image.open(image_file)
        width, height = img.size
        if width >= 512 and height >= 512:
            img = img.resize((512, 512))
        with BytesIO() as buffer:
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

        description = ""
        for response in generate(
            model='llava',
            prompt='Describe this image and include any notable details (include text in the image):',
            images=[image_bytes],
            stream=True
        ):
            description += response.get('response', '')

        # Add to DataFrame and save
        image_name = image_file.name if hasattr(image_file, 'name') else "uploaded_image"
        df.loc[len(df)] = [image_name, description]
        df.to_csv('image_descriptions.csv', index=False)

        return description
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "An error occurred while processing the image."

def extract_text_from_audio_llama(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        extracted_text = recognizer.recognize_google(audio)
        return extracted_text
    except sr.UnknownValueError:
        st.error("Speech recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Error with the speech recognition service: {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return ""

def process_uploaded_file_llama(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf_llama(uploaded_file)
        return text
    elif uploaded_file.type.startswith("image/"):
        description = describe_image_with_llava_llama(uploaded_file)
        return description
    elif uploaded_file.type.startswith("audio/"):
        transcription = extract_text_from_audio_llama(uploaded_file)
        return transcription
    else:
        st.error("Unsupported file type.")
        return ""

def split_and_summarize_text_llama(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    summaries = []
    for chunk in chunks[:5]:  # Limit summarization to the first 5 chunks for efficiency
        summary_prompt = f"Summarize this text:\n\n{chunk}"
        summary = llama_model.invoke(summary_prompt)
        summaries.append(summary.strip())

    return "\n".join(summaries), chunks

def create_faiss_store_llama(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    faiss_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return faiss_store

def search_relevant_context_llama(faiss_store, query):
    relevant_chunks = faiss_store.similarity_search(query, k=3)  # Top 3 results
    return "\n".join([chunk.page_content for chunk in relevant_chunks])

def extract_key_sections_llama(text):
    prompt = f"Extract the key sections (e.g., Title, Author, Abstract) from this text:\n\n{text}"
    key_sections = llama_model.invoke(prompt)
    return key_sections.strip()

def chat_with_llama_llama(context, user_input):
    try:
        prompt = f"""
        ### Context:
        {context}
        ###
        User: {user_input}
        ###
        Llama, please respond concisely and accurately based on the context above.
        """
        response = llama_model.invoke(prompt)
        return response.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

def run_llama_interface():
    st.title("Interactive Chat with Uploaded Content - Llama Model")
        
    # Download Chat History
    if st.sidebar.button("Download Llama Chat History"):
        if st.session_state.llama_chat_history:
            chat_history_text = "\n".join(
                [f"{sender}: {message}" for sender, message in st.session_state.llama_chat_history]
            )
            full_text = f"Key Sections:\n{st.session_state.llama_key_sections}\n\nChat History:\n{chat_history_text}"
            st.download_button(
                label="Your file is ready! Click me!",
                data=full_text,
                file_name="llama_chat_history.txt",
                mime="text/plain",
            )
        else:
            st.warning("No chat history to download.")

    # Sidebar for file upload
    st.sidebar.header("Llama Upload File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "jpg", "png", "wav"], key="llama_file_uploader")

    # Clear Chat History Button
    if st.sidebar.button("New Llama Session"):
        st.session_state.clear()
        st.session_state.llama_context = ""
        st.session_state.llama_chat_history = []
        st.session_state.llama_key_sections = ""
        st.session_state.llama_faiss_store = None
        st.session_state.llama_user_input = ""
        st.session_state.llama_uploaded_file = None
        st.sidebar.success("Chat history cleared.")
            
    # Ensure session states are initialized for Llama
    for key in ["llama_context", "llama_chat_history", "llama_key_sections", "llama_faiss_store", "llama_user_input"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key in ["llama_context", "llama_key_sections", "llama_user_input"] else []

    # Process uploaded file
    if uploaded_file:
        file_content = process_uploaded_file_llama(uploaded_file)
        
        if file_content:
            # Update session state with the processed content
            st.session_state.llama_context = file_content
            
            # Extract or process key sections only if not already available
            if "key_sections" not in st.session_state or not st.session_state.llama_key_sections:
                st.session_state.llama_key_sections = extract_key_sections_llama(file_content) if uploaded_file.type == "application/pdf" else file_content

            # Special handling for PDFs to create FAISS store
            if uploaded_file.type == "application/pdf":
                summarized_text, chunks = split_and_summarize_text_llama(file_content)
                faiss_store = create_faiss_store_llama(chunks)
                st.session_state.llama_faiss_store = faiss_store

            # Display the uploaded content
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            st.sidebar.success("Content loaded successfully!")
        # Handle unsupported file types
        else:
            st.error("Unsupported file type. Please upload a PDF, image, or audio file.")

    # Display Key Sections in the Sidebar
    st.sidebar.subheader("Llama Key Sections")
    if "llama_key_sections" in st.session_state:
        st.sidebar.markdown(st.session_state.llama_key_sections, unsafe_allow_html=True)

    # Prompt Input Box at Bottom Center
    st.markdown(
    """
    <style>
    .prompt-bar {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .streamlit-button {
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    if "llama_user_input" not in st.session_state:
        st.session_state.llama_user_input = ""  # Initialize user input state

    # User Input
    llama_user_input = st.text_input(
        "Llama Enter your message:",
        key="llama_user_input",
        value=st.session_state.llama_user_input,  # Bind value to session state
        placeholder="Type your message here...",
        help="Chat with Llama about the uploaded file.",
    )
    
    if st.button("Send Llama"):
        if st.session_state.llama_context and llama_user_input.strip():
            st.session_state.llama_chat_history.append(("User", llama_user_input))
            with st.spinner("Llama is generating a response..."):
                # Check if a FAISS store exists 
                if "faiss_store" in st.session_state and st.session_state.llama_faiss_store:
                    relevant_context = search_relevant_context_llama(st.session_state.llama_faiss_store, llama_user_input)
                else:
                    # Use the full context directly for images or other non-FAISS scenarios
                    relevant_context = st.session_state.llama_context
                
                # Generate response based on the context
                response = chat_with_llama_llama(relevant_context, llama_user_input)
            
            # Update chat history
            st.session_state.llama_chat_history.append(("Llama", response))
            llama_user_input = ""  # Reset input field
        else:
            st.warning("Please upload content and enter a valid message.")

        # Display Chat History
    st.subheader("Chat History - Llama")
    if st.session_state.llama_chat_history:
        # Add custom CSS for chat styling
        st.markdown(
            """
            <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 10px;
                max-width: 600px;
                margin: auto;
            }
            .chat-row {
                display: flex;
                align-items: flex-start;
                gap: 10px;
            }
            .chat-row.reverse {
                flex-direction: row-reverse;
            }
            .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: #ddd;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 20px;
            }
            .user-message, .ai-message {
                padding: 10px;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background-color: #d1e7dd; /* Light green */
                color: #0f5132;
            }
            .ai-message {
                background-color: #f8d7da; /* Light red */
                color: #842029;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Begin chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for sender, message in st.session_state.llama_chat_history:
            if sender == "User":
                st.markdown(
                    f"""
                    <div class="chat-row reverse">
                        <div class="avatar">🧑</div>
                        <div class="user-message">{message}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-row">
                        <div class="avatar">🦙</div>
                        <div class="ai-message">{message}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Start chatting by uploading a PDF and entering a message.")

# ==================== Cohere Model Interface ====================

# Functions for Cohere Model

def extract_pdf_text_cohere(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_image_text_cohere(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No text detected in the image."
    except Exception as e:
        return f"Error processing image: {e}"

def extract_audio_text_cohere(file):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data).strip()
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

def generate_response_cohere(context, user_input):
    try:
        if not context.strip():
            context = "You are an intelligent assistant. Provide clear, concise, and friendly responses to user queries."

        prompt = f"""
        {context}

        User Question: {user_input}

        AI Response:
        """
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def summarize_content_cohere(content):
    return generate_response_cohere(content, "Can you summarize this content?")

def save_audio_to_wav_cohere(audio_data, fs):
    audio_file = io.BytesIO()
    write(audio_file, fs, audio_data)
    audio_file.seek(0)
    return audio_file

def start_recording_cohere():
    fs = 44100
    st.session_state["recording_cohere"] = True
    st.session_state["audio_data_cohere"] = sd.rec(int(10 * fs), samplerate=fs, channels=1, dtype="int16")
    st.session_state["fs_cohere"] = fs

def stop_recording_cohere():
    sd.stop()
    st.session_state["recording_cohere"] = False
    st.success("Recording stopped. Processing audio...")

    # Process audio
    audio_file = save_audio_to_wav_cohere(st.session_state["audio_data_cohere"], st.session_state["fs_cohere"])
    transcript = extract_audio_text_cohere(audio_file)

    if transcript:
        # Append the transcript as user's message
        st.session_state["cohere_messages"].append({"role": "user", "content": transcript})

        # Build context from previous messages
        context = "\n".join(
            [
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in st.session_state["cohere_messages"][:-1]
            ]
        )

        # Generate response from Cohere
        response = generate_response_cohere(context, transcript)
        st.session_state["cohere_messages"].append({"role": "ai", "content": response})
    else:
        st.session_state["cohere_messages"].append({"role": "system", "content": "Sorry, I could not understand the audio."})

def save_conversation_cohere():
    if st.session_state["cohere_messages"]:
        # Get a custom name from the user
        custom_name = st.sidebar.text_input("Enter a name for this conversation:", key="custom_name_input_cohere")
        
        # Default to timestamp if no name provided
        if not custom_name.strip():
            custom_name = f"Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Format and save the conversation as JSON
        conversation_data = {
            "conversation_id": custom_name,
            "messages": st.session_state["cohere_messages"],
        }
        file_path = os.path.join("conversations", f"{custom_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        st.sidebar.success(f"Conversation saved as '{custom_name}'!")

def load_saved_conversation_cohere(selected_file):
    if selected_file:
        with open(os.path.join("conversations", selected_file), "r", encoding="utf-8") as f:
            conversation_data = json.load(f)
        st.session_state["cohere_messages"] = conversation_data["messages"]
        st.success(f"Loaded conversation: {conversation_data['conversation_id']}")

def run_cohere_interface():
    st.title("AI Chat with File and Audio Support - Cohere Model")

    # Initialize session state for Cohere
    if "cohere_messages" not in st.session_state:
        st.session_state.cohere_messages = []
    if "uploaded_pdfs_cohere" not in st.session_state:
        st.session_state.uploaded_pdfs_cohere = []
    if "uploaded_images_cohere" not in st.session_state:
        st.session_state.uploaded_images_cohere = []
    if "recording_cohere" not in st.session_state:
        st.session_state.recording_cohere = False
    if "audio_data_cohere" not in st.session_state:
        st.session_state.audio_data_cohere = None
    if "fs_cohere" not in st.session_state:
        st.session_state.fs_cohere = 44100
    if "user_input_cohere" not in st.session_state:
        st.session_state.user_input_cohere = ""

    st.sidebar.title("Cohere Upload and Save")

    # File uploader
    uploaded_files_cohere = st.sidebar.file_uploader("Upload Files (PDF, Images):", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="cohere_file_uploader")

    if uploaded_files_cohere:
        for file in uploaded_files_cohere:
            if file.type == "application/pdf" and file not in st.session_state.uploaded_pdfs_cohere:
                st.session_state.uploaded_pdfs_cohere.append(file)
                st.sidebar.success(f"Uploaded PDF: {file.name}")
            elif file.type.startswith("image/") and file not in st.session_state.uploaded_images_cohere:
                st.session_state.uploaded_images_cohere.append(file)
                st.sidebar.success(f"Uploaded Image: {file.name}")

    # Sidebar for viewing uploaded files
    st.sidebar.subheader("📄 View PDFs")
    for idx, pdf in enumerate(st.session_state.uploaded_pdfs_cohere):
        if st.sidebar.button(f"View PDF: {pdf.name}", key=f"view-pdf-cohere-{idx}"):
            content = extract_pdf_text_cohere(pdf)
            summary = summarize_content_cohere(content)
            st.session_state.cohere_messages.append({"role": "system", "content": f"📄 Summary: {summary}"})

    st.sidebar.subheader("🖼️ View Images")
    for idx, image in enumerate(st.session_state.uploaded_images_cohere):
        if st.sidebar.button(f"View Image: {image.name}", key=f"view-image-cohere-{idx}"):
            content = extract_image_text_cohere(image)
            summary = summarize_content_cohere(content)
            st.session_state.cohere_messages.append({"role": "system", "content": f"🖼️ Summary: {summary}"})

    # Load saved conversations
    st.sidebar.subheader("Load Saved Conversation")
    saved_files_cohere = [f for f in os.listdir("conversations") if f.endswith(".json")]
    selected_file_cohere = st.sidebar.selectbox("Select a conversation to load:", [""] + saved_files_cohere, key="cohere_selectbox")

    if st.sidebar.button("Load Selected Conversation Cohere"):
        load_saved_conversation_cohere(selected_file_cohere)

    # Save conversation button
    if st.sidebar.button("Save Conversation Cohere"):
        save_conversation_cohere()

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.cohere_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-row reverse"><div class="avatar">🧑</div><div class="user-message-cohere">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-row"><div class="avatar">🤖</div><div class="ai-message-cohere">{msg["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input box with "Enter to send" functionality
    def handle_enter_cohere():
        user_input = st.session_state["user_input_cohere"].strip()
        if user_input:
            st.session_state["cohere_messages"].append({"role": "user", "content": user_input})
            context = "\n".join(
                [
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in st.session_state["cohere_messages"][:-1]
                ]
            )
            response = generate_response_cohere(context, user_input)
            st.session_state["cohere_messages"].append({"role": "ai", "content": response})
            st.session_state["user_input_cohere"] = ""

    st.text_input("Type your message:", key="user_input_cohere", on_change=handle_enter_cohere, placeholder="Type here and press Enter to send...")

    # Recording buttons
    if not st.session_state["recording_cohere"]:
        if st.button("🎤 Start Recording Cohere"):
            start_recording_cohere()
    else:
        if st.button("⏹ Stop Recording Cohere"):
            stop_recording_cohere()

    # Download Chat History
    if st.sidebar.button("Download Cohere Chat History"):
        if st.session_state.cohere_messages:
            chat_history_text = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.cohere_messages]
            )
            full_text = f"Chat History:\n{chat_history_text}"
            st.download_button(
                label="Download Chat History",
                data=full_text,
                file_name="cohere_chat_history.txt",
                mime="text/plain",
            )
        else:
            st.warning("No chat history to download.")

# ==================== Main Application ====================

def main_app():
    # Set the unified page style
    set_page_style()

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.radio("Select a model to interact with:", ('Gemini Model', 'Llama Model', 'Cohere Model'))

    if model_choice == 'Gemini Model':
        run_gemini_interface()
    elif model_choice == 'Llama Model':
        run_llama_interface()
    elif model_choice == 'Cohere Model':
        run_cohere_interface()

if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
