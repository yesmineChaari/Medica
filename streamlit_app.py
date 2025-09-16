import streamlit as st
from model import generate_safe_answer
from audio_utils import record_audio, transcribe_audio, synthesize_speech
from TTS.api import TTS
import os
from appointment_booking.appointment_agent.graph import app_graph
from dotenv import load_dotenv
try:
    from langfuse.langchain import CallbackHandler
except Exception:
    CallbackHandler = None

load_dotenv()
@st.cache_resource
def load_rag_pipeline():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import Qdrant
    from langchain_ollama import OllamaLLM as Ollama

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    vectorstore = Qdrant.from_existing_collection(
        collection_name="medical_qa_bge_large_en",
        embedding=embedding_model,
        url="http://localhost:6333",
        api_key=None,
    )
    llm = Ollama(model="llama3", temperature=0.3)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.7}
    )
    return retriever, llm

retriever, rag_llm = load_rag_pipeline()
@st.cache_resource
def load_tts():
    model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)
    speaker = model.speakers[0]
    return model, speaker


@st.cache_resource
def load_langfuse():
    if CallbackHandler is None:
        return None
    kwargs = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "host": os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    }
    try:
        return CallbackHandler(**kwargs)
    except TypeError:
        return CallbackHandler()

langfuse_handler = load_langfuse()

tts_model, tts_speaker = load_tts()

st.set_page_config(page_title="MEDICA", page_icon="🩺")
import logging
logging.basicConfig(level=logging.WARNING)  
st.title("MEDICA - Your AI Medical Assistant")
st.markdown("Ask a medical question and get AI-powered responses based on trusted data.")

# --- Mode selection ---
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "qa"  # 'qa' or 'appointment'

st.sidebar.title("Chat Mode")
mode = st.sidebar.radio("Choose chat mode:", ("Medical Q&A", "Appointment Booking"))
if mode == "Medical Q&A":
    st.session_state.chat_mode = "qa"
else:
    st.session_state.chat_mode = "appointment"

# --- Medical Q&A Chatbot ---
if st.session_state.chat_mode == "qa":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    def handle_text_submit():
        user_input = st.session_state.text_input
        if user_input.strip() == "":
            return
        st.session_state.chat_history.append(("user", user_input))
        response = generate_safe_answer(user_input, retriever, rag_llm, langfuse_handler)
        st.session_state.chat_history.append(("bot", response))
        st.session_state.text_input = ""  # Clear the input safely here

    st.text_input("Type your question:", key="text_input", on_change=handle_text_submit)
    # Audio input section (inside the QA block)
    if st.button("🎙️ Ask by Voice"):
        with st.spinner("Recording your question..."):
            audio_path = record_audio(duration=5)
            transcription, _ = transcribe_audio(audio_path)
            st.success("Transcription complete!")
            st.session_state.chat_history.append(("user", transcription))
            response = generate_safe_answer(transcription, retriever, rag_llm, langfuse_handler)
            st.session_state.chat_history.append(("bot", response))

    st.markdown("---")
    for speaker, text in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"🧑‍💬 **You:** {text}")
        else:
            #st.markdown(f"🤖 **Bot:** {text}")
            st.markdown(f"🤖 **Bot:** {text}")
            audio_path = synthesize_speech(text, tts_model, tts_speaker)
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            os.unlink(audio_path)   # delete temp file immediately after reading
            st.audio(audio_data, format="audio/wav")

# --- Appointment Booking Chatbot ---
else:
    st.header("Book an Appointment (Conversational)")
    # Initialize appointment state and chat history
    if "appt_state" not in st.session_state:
        st.session_state.appt_state = {
            "user_messages": [],
            "bot_messages": [],
            "greeting_sent": False,
            "last_user_intent": "",
            "date": None,
            "time": None,
            "email": None,
            "confirmed": None,
            "awaiting_user_response": False,
        }
    if "appt_chat_history" not in st.session_state:
        st.session_state.appt_chat_history = []
    if "appt_text_input" not in st.session_state:
        st.session_state.appt_text_input = ""

    # On first entry, greet the user
    if not st.session_state.appt_state["greeting_sent"] and not st.session_state.appt_chat_history:
        greeting = "Hello! I'm here to help you book an appointment. How can I assist you today?"
        st.session_state.appt_chat_history.append(("bot", greeting))
        st.session_state.appt_state["greeting_sent"] = True

    def handle_appt_text_submit():
        user_input = st.session_state.appt_text_input
        if user_input.strip() == "":
            return
        st.session_state.appt_chat_history.append(("user", user_input))
        st.session_state.appt_state["user_messages"].append(user_input)
        st.session_state.appt_state["awaiting_user_response"] = False
        # Call the appointment booking graph
        config = {"configurable": {"thread_id": "streamlit-session"}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        result = app_graph.invoke(st.session_state.appt_state, config=config)
        st.session_state.appt_state = result
        # Add bot messages to chat history
        for msg in result.get("bot_messages", []):
            if msg:
                st.session_state.appt_chat_history.append(("bot", msg))
        st.session_state.appt_state["bot_messages"] = []
        st.session_state.appt_text_input = ""  # Clear input

    st.text_input("Type your message:", key="appt_text_input", on_change=handle_appt_text_submit)
    if st.button("🎙️ Speak to Book"):
        with st.spinner("Recording your message..."):
            audio_path = record_audio(duration=5)
            transcription, _ = transcribe_audio(audio_path)
            st.success("Transcription complete!")
            st.session_state.appt_chat_history.append(("user", transcription))
            st.session_state.appt_state["user_messages"].append(transcription)
            st.session_state.appt_state["awaiting_user_response"] = False

            # Call the appointment booking graph
            config = {"configurable": {"thread_id": "streamlit-session"}}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]
            result = app_graph.invoke(st.session_state.appt_state, config=config)
            st.session_state.appt_state = result

            # Add bot messages to chat history
            for msg in result.get("bot_messages", []):
                if msg:
                    st.session_state.appt_chat_history.append(("bot", msg))
            st.session_state.appt_state["bot_messages"] = []

    st.markdown("---")
    for speaker, text in st.session_state.appt_chat_history:
        if speaker == "user":
            st.markdown(f"🧑‍💬 **You:** {text}")
        else:
            st.markdown(f"🤖 **Bot:** {text}")
            audio_path = synthesize_speech(text, tts_model, tts_speaker)
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            os.unlink(audio_path)   # delete temp file immediately after reading
            st.audio(audio_data, format="audio/wav")
    # Show confirmation if appointment is booked
    appt = st.session_state.appt_state
    if appt.get("confirmed") and appt.get("date") and appt.get("time"):
        st.balloons()
        st.info(f"Your appointment has been booked successfully!\nDate: {appt['date']}\nTime: {appt['time']}\nConfirmation will be sent to: {appt.get('email','')}")
