import streamlit as st
import logging
import os
from model import (
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    generate_safe_answer,
)
from audio_utils import (
    record_audio,
    start_audio_recording,
    stop_audio_recording,
    transcribe_audio,
    synthesize_speech,
)
from TTS.api import TTS
from appointment_booking.appointment_agent.graph import app_graph
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
try:
    from langfuse.langchain import CallbackHandler
except Exception:
    CallbackHandler = None

logger = logging.getLogger(__name__)
HISTORY_WINDOW = 6

load_dotenv()
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Medica")


@st.cache_resource
def load_rag_pipeline():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import Qdrant
    from langchain_ollama import OllamaLLM as Ollama

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Qdrant.from_existing_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
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
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    kwargs = {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    }
    try:
        return CallbackHandler(**kwargs)
    except Exception as exc:
        logger.warning("Langfuse tracing is disabled: %s", exc)
        return None

langfuse_handler = load_langfuse()

tts_model, tts_speaker = load_tts()

st.title("Medica - Your AI Medical Assistant")
st.markdown("Ask a medical question and get AI-powered responses based on trusted data.")

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "qa" 

st.sidebar.title("Chat Mode")
mode = st.sidebar.radio("Choose chat mode:", ("Medical Q&A", "Appointment Booking"))
if mode == "Medical Q&A":
    st.session_state.chat_mode = "qa"
else:
    st.session_state.chat_mode = "appointment"

if st.session_state.chat_mode == "qa":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "voice_recording" not in st.session_state:
        st.session_state.voice_recording = False
    if "voice_recorder" not in st.session_state:
        st.session_state.voice_recorder = None
    if "voice_notice" not in st.session_state:
        st.session_state.voice_notice = None

    def cache_bot_audio(msg_index, response):
        audio_path = None
        try:
            audio_path = synthesize_speech(response, tts_model, tts_speaker)
            with open(audio_path, "rb") as f:
                st.session_state.audio_cache[msg_index] = f.read()
        except Exception:
            st.session_state.audio_cache[msg_index] = None
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    def build_rag_chat_history():
        history = []
        for speaker, text in st.session_state.chat_history[-HISTORY_WINDOW:]:
            if speaker == "user":
                history.append(HumanMessage(content=text))
            else:
                history.append(AIMessage(content=text))
        return history

    def handle_text_submit():
        user_input = st.session_state.text_input
        if user_input.strip() == "":
            return
        history = build_rag_chat_history()
        st.session_state.chat_history.append(("user", user_input))
        response = generate_safe_answer(user_input, retriever, rag_llm, history, langfuse_handler)
        st.session_state.chat_history.append(("bot", response))
        msg_index = len(st.session_state.chat_history) - 1
        cache_bot_audio(msg_index, response)
        st.session_state.text_input = ""

    st.text_input("Type your question:", key="text_input", on_change=handle_text_submit)
    voice_notice = st.session_state.voice_notice
    if voice_notice:
        notice_type, notice_text = voice_notice
        if notice_type == "success":
            st.success(notice_text)
        elif notice_type == "warning":
            st.warning(notice_text)
        else:
            st.error(notice_text)
        st.session_state.voice_notice = None

    if not st.session_state.voice_recording:
        if st.button("Ask by Voice", key="start_voice_recording"):
            try:
                st.session_state.voice_recorder = start_audio_recording()
                st.session_state.voice_recording = True
            except Exception as exc:
                st.session_state.voice_recorder = None
                st.session_state.voice_recording = False
                st.session_state.voice_notice = ("error", f"Could not start recording: {exc}")
            st.rerun()
    else:
        st.info("Recording your question...")
        if st.button("Stop Recording", key="stop_voice_recording"):
            recorder = st.session_state.voice_recorder
            st.session_state.voice_recorder = None
            st.session_state.voice_recording = False
            audio_path = None
            try:
                with st.spinner("Transcribing your question..."):
                    audio_path = stop_audio_recording(recorder)
                    transcription, _ = transcribe_audio(audio_path)
                transcription = transcription.strip()
                if transcription:
                    history = build_rag_chat_history()
                    st.session_state.chat_history.append(("user", transcription))
                    response = generate_safe_answer(transcription, retriever, rag_llm, history, langfuse_handler)
                    st.session_state.chat_history.append(("bot", response))
                    msg_index = len(st.session_state.chat_history) - 1
                    cache_bot_audio(msg_index, response)
                    st.session_state.voice_notice = ("success", "Transcription complete!")
                else:
                    st.session_state.voice_notice = ("warning", "No speech detected. Please try again.")
            except Exception as exc:
                st.session_state.voice_notice = ("error", f"Voice recording failed: {exc}")
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            st.rerun()

    st.markdown("---")
    for i, (speaker, text) in enumerate(st.session_state.chat_history):
        if speaker == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")
            audio_bytes = st.session_state.audio_cache.get(i)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

else:
    st.header("Book an Appointment (Conversational)")
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
        config = {"configurable": {"thread_id": "streamlit-session"}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        try:
            result = app_graph.invoke(st.session_state.appt_state, config=config)
            st.session_state.appt_state = result
            for msg in result.get("bot_messages", []):
                if msg:
                    st.session_state.appt_chat_history.append(("bot", msg))
            st.session_state.appt_state["bot_messages"] = []
        except Exception as e:
            logger.error("Appointment graph failed: %s", e)
            st.session_state.appt_chat_history.append((
                "bot",
                "I encountered an error processing your request. Please try again or contact the clinic directly."
            ))
        st.session_state.appt_text_input = ""

    st.text_input("Type your message:", key="appt_text_input", on_change=handle_appt_text_submit)
    if st.button("Speak to Book"):
        with st.spinner("Recording your message..."):
            audio_path = record_audio(duration=5)
            transcription, _ = transcribe_audio(audio_path)
            st.success("Transcription complete!")
            st.session_state.appt_chat_history.append(("user", transcription))
            st.session_state.appt_state["user_messages"].append(transcription)
            st.session_state.appt_state["awaiting_user_response"] = False

            config = {"configurable": {"thread_id": "streamlit-session"}}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]
            try:
                result = app_graph.invoke(st.session_state.appt_state, config=config)
                st.session_state.appt_state = result

                for msg in result.get("bot_messages", []):
                    if msg:
                        st.session_state.appt_chat_history.append(("bot", msg))
                st.session_state.appt_state["bot_messages"] = []
            except Exception as e:
                logger.error("Appointment graph failed: %s", e)
                st.session_state.appt_chat_history.append((
                    "bot",
                    "I encountered an error processing your request. Please try again or contact the clinic directly."
                ))

    st.markdown("---")
    for speaker, text in st.session_state.appt_chat_history:
        if speaker == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")
            audio_path = synthesize_speech(text, tts_model, tts_speaker)
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            os.unlink(audio_path)
            st.audio(audio_data, format="audio/wav")
    appt = st.session_state.appt_state
    if appt.get("confirmed") and appt.get("date") and appt.get("time"):
        st.balloons()
        st.info(f"Your appointment has been booked successfully!\nDate: {appt['date']}\nTime: {appt['time']}\nConfirmation will be sent to: {appt.get('email','')}")
