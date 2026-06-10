# Tea Vision AI

AI-powered platform for **tea leaf disease detection** and **tea cultivation assistance**. Upload a photo of a tea leaf to identify diseases with a deep learning model, or chat with an expert assistant trained on tea farming documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

---

## Features

| Feature | Description |
|---------|-------------|
| **Leaf Disease Detection** | Upload a leaf image and get AI classification with confidence scores and a probability chart |
| **Treatment Guidance** | Symptoms, severity, and recommended actions for each detected disease |
| **Tea Cultivation Chatbot** | RAG-powered assistant for questions about diseases, fertilizers, and best practices |
| **Green-Themed UI** | Modern Streamlit frontend designed for tea farmers |
| **REST API** | Flask backend with `/predict` and `/chat` endpoints |

---

## Tech Stack

- **Frontend:** Streamlit
- **Disease Model:** DenseNet201 transfer learning (`final_tea_model2.h5`)
- **Chatbot:** LangChain + ChromaDB + OpenAI GPT-3.5
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Backend API:** Flask + Flask-CORS
- **CV / ML:** TensorFlow, Keras, OpenCV, NumPy

---

## Project Structure

```
tea-vision-ai/
├── ai-service/
│   ├── streamlit_app.py          # Streamlit frontend (main UI)
│   ├── app.py                    # Flask REST API
│   ├── predict_cli.py            # CLI wrapper for disease prediction
│   ├── chatBot/
│   │   ├── tea_chatbot.py        # RAG chatbot logic
│   │   ├── chatBot.ipynb         # Build ChromaDB knowledge base
│   │   ├── chroma_db/            # Vector store (generated)
│   │   └── Resources/            # PDF knowledge documents
│   ├── DiseasePrediction/
│   │   ├── prediction Model2/
│   │   │   ├── prediction2.py    # DenseNet disease predictor
│   │   │   └── final_tea_model2.h5   # Trained model weights (required)
│   │   ├── data-preprocessing.ipynb
│   │   └── model_creation (1).ipynb
│   └── services/
│       ├── predictor.py          # Loads prediction2 module
│       ├── disease_info.py       # Disease metadata
│       └── env_config.py         # Environment variable helpers
├── .streamlit/
│   └── config.toml               # Green theme configuration
├── requirements.txt
├── run_streamlit.bat             # Windows launcher (uses project venv)
└── README.md
```

---

## Supported Disease Classes

| # | Disease |
|---|---------|
| 0 | Tea algal leaf spot |
| 1 | Brown Blight |
| 2 | Gray Blight |
| 3 | Helopeltis |
| 4 | Red spider |
| 5 | Green mirid bug |
| 6 | Healthy leaf |

---

## Prerequisites

- Python 3.10 or 3.11
- Windows, macOS, or Linux
- OpenAI API key (for chatbot)
- Trained model file: `final_tea_model2.h5`
- ~2 GB free disk space (TensorFlow + dependencies)

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd tea-vision-ai
```

### 2. Create a virtual environment

```bash
python -m venv env
```

**Windows (activate):**
```bash
env\Scripts\activate
```

**macOS / Linux (activate):**
```bash
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the disease model

Place your trained model weights at:

```
ai-service/DiseasePrediction/prediction Model2/final_tea_model2.h5
```

> This file is required for disease detection. It is not included in the repository (gitignored).

### 5. Configure the chatbot

Create `ai-service/chatBot/.env`:

```env
API_KEY=your-openai-api-key-here
```

> Use a valid key from [OpenAI API Keys](https://platform.openai.com/api-keys).  
> If you also have `ai-service/.env`, the chatbot key in `chatBot/.env` takes priority.

### 6. Build the chatbot knowledge base (first time only)

Open and run `ai-service/chatBot/chatBot.ipynb` to ingest PDF documents from `chatBot/Resources/` and create the `chroma_db` vector store.

---

## Running the Application

### Streamlit UI (recommended)

**Windows — double-click or run:**
```bash
run_streamlit.bat
```

**Manual start (from project root):**
```bash
env\Scripts\streamlit run ai-service/streamlit_app.py    # Windows
env/bin/streamlit run ai-service/streamlit_app.py        # macOS / Linux
```

Open your browser at: **http://localhost:8501**

#### Streamlit pages

| Page | Description |
|------|-------------|
| **Home** | Overview, supported diseases, how it works |
| **Disease Detection** | Upload a leaf image and analyze |
| **Tea Assistant** | Chat with the cultivation expert |

### Flask REST API (optional)

From the `ai-service` directory:

```bash
cd ai-service
python app.py
```

API runs at: **http://localhost:5000**

---

## API Reference

### `POST /predict`

Upload a tea leaf image for disease classification.

**Request:** `multipart/form-data` with field `image`

**Example (curl):**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@leaf.jpg"
```

**Response:**
```json
{
  "success": true,
  "predicted_disease": "Red spider",
  "confidence": 59.32
}
```

---

### `POST /chat`

Ask the tea cultivation assistant a question.

**Request:** `application/json`

```json
{
  "question": "What causes Brown Blight in tea?"
}
```

**Response:**
```json
{
  "success": true,
  "question": "What causes Brown Blight in tea?",
  "answer": "..."
}
```

---

## Disease Prediction Details

- **Model:** DenseNet201 (`final_tea_model2.h5`)
- **Inference script:** `prediction Model2/prediction2.py`
- **Input size:** 128 × 128 RGB
- **Preprocessing:** BGR → RGB, resize, normalize to `[0, 1]`

Standalone CLI test:

```bash
cd ai-service
python predict_cli.py path/to/leaf.jpg
```

---

## Troubleshooting

### `No module named 'tensorflow'`

Run the app with the project virtual environment:

```bash
run_streamlit.bat
```

The app automatically falls back to the `env` venv for predictions when TensorFlow is missing from the active Python.

### `401 Incorrect API key` (chatbot)

1. Add a valid OpenAI key to `ai-service/chatBot/.env`
2. Remove or update any invalid key in `ai-service/.env`
3. Restart the app

### `Prediction failed: Expecting value: line 1 column 1`

This was caused by TensorFlow log output mixing with JSON. Ensure you are on the latest code — `predict_cli.py` now outputs clean JSON only.

### Same prediction / ~14% confidence for every image

Use **`final_tea_model2.h5`** (DenseNet). The older `final_tea_model.h5` CNN model produces near-uniform outputs and should not be used.

### Chatbot says knowledge base not found

Run `chatBot.ipynb` to build `ai-service/chatBot/chroma_db/`.

### Model weights not found

Confirm the file exists at:
```
ai-service/DiseasePrediction/prediction Model2/final_tea_model2.h5
```

---

## Training (Notebooks)

| Notebook | Purpose |
|----------|---------|
| `DiseasePrediction/data-preprocessing.ipynb` | Prepare image dataset |
| `DiseasePrediction/model_creation (1).ipynb` | Train CNN model (v1) |
| `DiseasePrediction/prediction Model2/data-preprocessing.ipynb` | Preprocess data for DenseNet |
| `chatBot/chatBot.ipynb` | Build RAG chatbot knowledge base |

Dataset folder: `teaLeafBD/` (not included in repo).

---

## Environment Variables

| Variable | Location | Purpose |
|----------|----------|---------|
| `API_KEY` | `ai-service/chatBot/.env` | OpenAI API key for chatbot |

---

## License

This project is for educational and research purposes. Check with your institution or team for licensing details.

---

## Contributors

Tea Vision AI — Smart farming for tea growers.
