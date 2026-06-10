import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure ai-service is on the path when launched from project root
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VENV_PYTHON = PROJECT_ROOT / "env" / "Scripts" / "python.exe"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.disease_info import CLASS_NAMES, DISEASE_INFO
from services.env_config import get_openai_api_key, load_app_env

load_app_env()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tea Vision AI",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — green tea-farm aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit header/footer clutter */
#MainMenu, footer, header { visibility: hidden; }

/* Hero banner */
.tea-hero {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 45%, #43A047 100%);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(27, 94, 32, 0.25);
}
.tea-hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: white !important;
}
.tea-hero p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 0;
    color: #E8F5E9 !important;
}

/* Feature cards */
.feature-card {
    background: white;
    border: 1px solid #C8E6C9;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    height: 100%;
    box-shadow: 0 2px 12px rgba(46, 125, 50, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(46, 125, 50, 0.15);
}
.feature-card .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-card h3 { color: #1B5E20; margin: 0 0 0.4rem 0; font-size: 1.1rem; }
.feature-card p  { color: #4E6B4E; margin: 0; font-size: 0.9rem; line-height: 1.5; }

/* Result panel */
.result-panel {
    background: linear-gradient(160deg, #E8F5E9, #FFFFFF);
    border: 2px solid #A5D6A7;
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-top: 1rem;
}
.result-panel.healthy { border-color: #66BB6A; }
.result-panel.disease { border-color: #EF9A9A; background: linear-gradient(160deg, #FFF8E1, #FFFFFF); }

.confidence-bar-bg {
    background: #E0E0E0;
    border-radius: 10px;
    height: 14px;
    overflow: hidden;
    margin: 0.6rem 0;
}
.confidence-bar-fill {
    background: linear-gradient(90deg, #43A047, #2E7D32);
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s ease;
}

/* Disease reference chips */
.disease-chip {
    display: inline-block;
    background: #F1F8E9;
    border: 1px solid #C8E6C9;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.82rem;
    color: #2E7D32;
}

/* Chat bubbles */
.chat-user {
    background: #2E7D32;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0;
    max-width: 85%;
    margin-left: auto;
}
.chat-bot {
    background: white;
    border: 1px solid #C8E6C9;
    color: #1B3A1B;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
}
[data-testid="stSidebar"] * { color: #E8F5E9 !important; }
[data-testid="stSidebar"] .stRadio label { font-weight: 500; }

/* Upload zone hint */
.upload-hint {
    text-align: center;
    color: #558B2F;
    padding: 1rem;
    border: 2px dashed #A5D6A7;
    border-radius: 14px;
    background: #F9FBE7;
    margin-bottom: 1rem;
}

/* Stat badges */
.stat-badge {
    background: rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    display: inline-block;
    margin-right: 0.8rem;
}
.stat-badge .num { font-size: 1.6rem; font-weight: 700; }
.stat-badge .lbl { font-size: 0.75rem; opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Lazy-loaded backends
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading disease detection model…")
def load_predictor():
    from services.predictor import predict_disease_detailed

    return predict_disease_detailed


@st.cache_resource(show_spinner="Initializing tea assistant…")
def load_chatbot():
    from chatBot.tea_chatbot import ask_question

    return ask_question


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_subprocess_json(stdout: str) -> dict:
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("Prediction subprocess returned empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"Invalid prediction output: {text[:200]}")


def run_prediction(image_path: str) -> dict:
    """Run prediction in-process or via project venv subprocess."""
    use_venv = VENV_PYTHON.exists() and (
        not tensorflow_available()
        or Path(sys.executable).resolve() != VENV_PYTHON.resolve()
    )

    if not use_venv:
        return load_predictor()(image_path)

    proc = subprocess.run(
        [str(VENV_PYTHON), str(APP_DIR / "predict_cli.py"), image_path],
        capture_output=True,
        text=True,
        cwd=str(APP_DIR),
        env={
            **os.environ,
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        },
    )

    if proc.returncode != 0:
        try:
            err = _parse_subprocess_json(proc.stdout)
            if "error" in err:
                raise RuntimeError(err["error"])
        except (json.JSONDecodeError, RuntimeError):
            pass
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or "Prediction failed"
        )

    data = _parse_subprocess_json(proc.stdout)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def model_weights_exist() -> bool:
    from services.predictor import model_exists

    return model_exists()


def chatbot_ready() -> bool:
    chroma = APP_DIR / "chatBot" / "chroma_db"
    return chroma.exists() and any(chroma.iterdir())


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🍵 Tea Vision AI")
    st.markdown("*Smart farming for tea growers*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Disease Detection", "💬 Tea Assistant"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("##### System Status")
    model_ok = model_weights_exist()
    chat_ok = chatbot_ready()
    tf_ok = tensorflow_available() or VENV_PYTHON.exists()
    st.markdown(f"{'🟢' if model_ok else '🔴'} Disease model weights")
    st.markdown(f"{'🟢' if tf_ok else '🔴'} TensorFlow runtime")
    st.markdown(f"{'🟢' if chat_ok else '🔴'} Chat knowledge base")
    if not tensorflow_available() and VENV_PYTHON.exists():
        st.caption("Using project `env` for predictions")
    if model_ok:
        try:
            from services.predictor import get_model_path

            st.caption(f"Model: `{get_model_path().name}`")
        except Exception:
            pass
    else:
        st.caption("Place `final_tea_model2.h5` in `DiseasePrediction/prediction Model2/`")
    if not chat_ok:
        st.caption("Run chatBot notebook to build `chroma_db`")
    st.divider()
    st.caption("Powered by TensorFlow & LangChain")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="tea-hero">
            <h1>🍵 Tea Vision AI</h1>
            <p>AI-powered tea leaf disease detection and expert cultivation guidance —
            helping farmers protect their crops with confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="stat-badge"><div class="num">7</div><div class="lbl">Disease Classes</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="stat-badge"><div class="num">128²</div><div class="lbl">Pixel Analysis</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stat-badge"><div class="num">RAG</div><div class="lbl">Expert Chatbot</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon">🔬</div>
                <h3>Disease Detection</h3>
                <p>Upload a photo of a tea leaf and get instant AI classification
                with confidence scores and treatment recommendations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon">💬</div>
                <h3>Tea Cultivation Assistant</h3>
                <p>Ask questions about tea diseases, fertilizers, and best practices.
                Powered by a knowledge base built from expert tea farming documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Supported Disease Classes")
    chips = " ".join(
        f'<span class="disease-chip">{name}</span>' for name in CLASS_NAMES
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("### How It Works")
    st.markdown(
        """
        1. **Upload** a clear photo of a tea leaf (JPEG or PNG)
        2. **Analyze** with our deep learning CNN model trained on thousands of leaf images
        3. **Get results** including disease name, confidence, symptoms & treatment tips
        4. **Chat** with the AI assistant for follow-up questions on fertilizers and control measures
        """
    )


# ---------------------------------------------------------------------------
# DISEASE DETECTION
# ---------------------------------------------------------------------------
elif page == "🔬 Disease Detection":
    st.markdown(
        """
        <div class="tea-hero" style="padding:1.6rem 2rem;">
            <h1 style="font-size:1.8rem;">🔬 Leaf Disease Detection</h1>
            <p>Upload a tea leaf image for instant AI-powered disease classification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_weights_exist():
        st.error(
            "**Model weights not found.** Place `final_tea_model2.h5` inside "
            "`ai-service/DiseasePrediction/prediction Model2/` and restart the app."
        )
        st.stop()

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.markdown(
            '<div class="upload-hint">📷 Drag & drop or click to upload a tea leaf image</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        predict_btn = st.button(
            "🔍 Analyze Leaf",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

        with st.expander("📋 Tips for best results"):
            st.markdown(
                """
                - Use a **well-lit** photo with the leaf filling most of the frame
                - Avoid blurry or heavily shadowed images
                - Single leaf per image works best
                - Supported formats: JPG, PNG, WEBP
                """
            )

    with col_preview:
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
        else:
            st.info("Preview will appear here after you upload an image.")

    if predict_btn and uploaded:
        with st.spinner("Analyzing leaf patterns…"):
            try:
                suffix = Path(uploaded.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                result = run_prediction(tmp_path)
                os.unlink(tmp_path)

                disease = result["disease"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]
                model_used = result.get("model", "unknown")

                info = DISEASE_INFO.get(disease, {})
                is_healthy = disease == "Healthy leaf"
                panel_class = "healthy" if is_healthy else "disease"
                icon = info.get("icon", "🍃")
                conf_pct = confidence * 100

                st.markdown(
                    f"""
                    <div class="result-panel {panel_class}">
                        <h2 style="color:#1B5E20; margin-top:0;">
                            {icon} {disease}
                        </h2>
                        <p style="color:#4E6B4E; margin:0.2rem 0 0.8rem;">
                            {'Great news — this leaf appears healthy!' if is_healthy
                             else 'Disease detected — review recommendations below.'}
                        </p>
                        <strong>Confidence: {conf_pct:.1f}%</strong>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width:{conf_pct:.1f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.markdown("#### Symptoms")
                    st.write(info.get("symptoms", "No additional information available."))
                with det_col2:
                    st.markdown("#### Recommended Action")
                    st.write(info.get("treatment", "Consult a local agronomist."))

                severity = info.get("severity", "Unknown")
                sev_color = {"None": "green", "Moderate": "orange", "High": "red"}.get(
                    severity, "blue"
                )
                st.markdown(f"**Severity:** :{sev_color}[{severity}]")
                st.caption(f"Model used: `{model_used}`")

                st.markdown("#### All Class Probabilities")
                sorted_probs = dict(
                    sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                )
                st.bar_chart(sorted_probs)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------------------------
elif page == "💬 Tea Assistant":
    st.markdown(
        """
        <div class="tea-hero" style="padding:1.6rem 2rem;">
            <h1 style="font-size:1.8rem;">💬 Tea Cultivation Assistant</h1>
            <p>Ask about tea diseases, fertilizers, pest control, and farming best practices</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your tea cultivation assistant. "
                    "Ask me about tea leaf diseases, fertilizer schedules, "
                    "pest management, or general farming practices. 🍵"
                ),
            }
        ]

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🍵 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Suggested prompts
    st.markdown("**Quick questions:**")
    suggestions = st.columns(3)
    prompts = [
        "What causes Brown Blight in tea?",
        "Recommended NPK fertilizer for tea?",
        "How to control red spider mites?",
    ]
    for col, prompt in zip(suggestions, prompts):
        with col:
            if st.button(prompt, use_container_width=True, key=f"sug_{prompt[:20]}"):
                st.session_state.pending_prompt = prompt

    # Chat input
    user_input = st.chat_input("Type your question about tea farming…")

    prompt = user_input or st.session_state.pop("pending_prompt", None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not chatbot_ready():
            answer = (
                "The knowledge base is not set up yet. "
                "Please run the `chatBot.ipynb` notebook to build the ChromaDB "
                "vector store, and ensure your OpenAI API key is in `.env`."
            )
        elif not get_openai_api_key():
            answer = (
                "OpenAI API key not found. "
                "Add `API_KEY=your-key` to `ai-service/chatBot/.env` and restart the app."
            )
        else:
            with st.spinner("Thinking…"):
                try:
                    ask_question = load_chatbot()
                    answer = ask_question(prompt)
                except Exception as exc:
                    err = str(exc)
                    if "401" in err or "invalid_api_key" in err or "Incorrect API key" in err:
                        answer = (
                            "Your OpenAI API key is invalid or expired. "
                            "Update `API_KEY` in `ai-service/chatBot/.env` with a valid key from "
                            "https://platform.openai.com/api-keys, then restart the app."
                        )
                    else:
                        answer = f"Sorry, I encountered an error: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("🗑️ Clear conversation", type="secondary"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
