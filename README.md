# 📝 End-to-End Text Summarizer (Pegasus + MLOps)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, production-grade **Abstractive Text Summarization** system designed to condense messy, multi-person chat logs and dialogues into crisp, accurate summaries. Built using Google's **Pegasus (`google/pegasus-cnn_dailymail`)** transformer model fine-tuned on the **SAMSum** conversational dataset, this project implements a modular **MLOps pipeline** from data ingestion to model evaluation, served via a **FastAPI** backend with a mounted **Gradio** web interface.

---

## 📸 Project Demo

<img width="1917" height="898" alt="Interactive Gradio Web UI" src="https://github.com/user-attachments/assets/e547077b-03eb-4a23-b526-65dc9c99cb3b" />

*Real-time dialogue summarization running on the mounted Gradio UI and FastAPI backend.*

---

## ✨ Key Highlights

- **Modular MLOps Architecture:** Strictly follows the *Separation of Concerns* design pattern—decoupling YAML configurations, data entities, core components, and execution pipelines inside a reusable `src/` Python package.
- **Custom Fine-Tuned Seq2Seq Model:** Fine-tunes the `pegasus-cnn_dailymail` architecture on conversational text (`SAMSum` dataset) using Hugging Face `Trainer` and PyTorch with local GPU acceleration.
- **Hybrid Serving Layer:** Combines an enterprise REST API (`FastAPI`) with an intuitive user interface (`Gradio`) mounted on the same Uvicorn server instance.
- **Centralized Logging & Exception Tracking:** Automatically captures timestamped execution traces across all pipeline stages into `logs/running_logs.log` and standard output.
- **Reproducible & Container-Ready:** Configured with `params.yaml`, `config.yaml`, `setup.py`, and a `Dockerfile` for seamless environment replication and CI/CD integration.

---

## 🏗️ MLOps Pipeline Architecture

The training and evaluation lifecycle is broken down into **5 modular stages** orchestrated by `main.py`:

1. **Stage 01 — Data Ingestion:** Downloads the raw conversational dataset archive and extracts it into structured artifact directories.
2. **Stage 02 — Data Validation:** Verifies the integrity and presence of all required dataset splits (`train`, `test`, `validation`) before compute-heavy processing begins.
3. **Stage 03 — Data Transformation:** Tokenizes raw dialogues and target summaries using the Pegasus `AutoTokenizer`, converting text into PyTorch-ready feature tensors (`input_ids`, `attention_mask`, `labels`).
4. **Stage 04 — Model Trainer:** Loads the pre-trained Pegasus weights and fine-tunes the sequence-to-sequence model on the transformed dataset using hyperparameters defined in `params.yaml`.
5. **Stage 05 — Model Evaluation:** Evaluates the fine-tuned checkpoint on the test split and computes standard NLP summarization metrics ($\text{ROUGE-1}$, $\text{ROUGE-2}$, $\text{ROUGE-L}$, and $\text{ROUGE-Lsum}$).

### Code Execution Flow
Every pipeline stage adheres to the following internal workflow:

$$\text{config.yaml} \rightarrow \text{params.yaml} \rightarrow \text{entity} \rightarrow \text{configuration.py} \rightarrow \text{components} \rightarrow \text{pipeline} \rightarrow \text{main.py}$$

---

## 📂 Repository Structure

```text
Text-Summarizer/
├── .github/
│   └── workflows/               # CI/CD GitHub Actions workflows
├── config/
│   └── config.yaml              # Directory paths & data source URLs
├── research/
│   └── trials.ipynb             # Jupyter sandbox for experimentation
├── src/
│   └── textSummarizer/
│       ├── components/          # Core logic for Ingestion, Validation, Transformation, Training & Evaluation
│       ├── config/
│       │   └── configuration.py # ConfigurationManager linking YAML files to dataclasses
│       ├── constants/           # Immutable file paths (CONFIG_FILE_PATH, PARAMS_FILE_PATH)
│       ├── entity/              # Python @dataclass definitions for type-safe configs
│       ├── logging/             # Centralized FileHandler & StreamHandler logger
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── prediction.py    # Inference pipeline for real-time summarization
│       └── utils/
│           └── common.py        # Helper functions (read_yaml, create_directories, etc.)
├── app.py                       # FastAPI server + mounted Gradio frontend UI
├── main.py                      # Master script to run all 5 training pipeline stages
├── params.yaml                  # Tunable model training & generation hyperparameters
├── requirements.txt             # Project dependencies
├── setup.py                     # Local package installation script
├── template.py                  # Automated project scaffolding script
└── Dockerfile                   # Containerization blueprint
```

---

## 🚀 Getting Started (Local Setup)

### Prerequisites
- Python $3.10+$
- Git
- *(Optional but recommended)* NVIDIA GPU with CUDA support for faster training and inference

### 1. Clone the Repository
```bash
git clone https://github.com/TrijalSharma/Text-Summarizer.git
cd Text-Summarizer
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv text_env
.\text_env\Scripts\activate

# Linux / macOS
python3 -m venv text_env
source text_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline *(Required on first run to generate model artifacts)*
To execute Data Ingestion $\rightarrow$ Validation $\rightarrow$ Transformation $\rightarrow$ Training $\rightarrow$ Evaluation and save the fine-tuned model weights into `artifacts/`:
```bash
python main.py
```

### 5. Launch the Web Application
Start the Uvicorn server hosting both the Gradio UI and FastAPI endpoints:
```bash
python app.py
```
- **Interactive Gradio Web App:** Open `http://127.0.0.1:8000/` in your browser.
- **FastAPI Swagger UI:** Open `http://127.0.0.1:8000/docs` to test the REST API.

---

## 🌐 Exposing the App Publicly (On-Demand via Ngrok)

To share a live demo link on mobile devices or during presentations without cloud memory bottlenecks:

1. Keep `python app.py` running in your primary terminal.
2. Open a second terminal and start the tunnel:
   ```bash
   ngrok http 8000
   ```
3. Copy the generated `https://xxxx.ngrok-free.app` forwarding link to access the summarizer from anywhere.

---

## 📊 Model Evaluation & Results

Upon completing **Stage 05**, the evaluation pipeline calculates $\text{ROUGE}$ metrics on the test dataset and saves the report to `artifacts/model_evaluation/metrics.csv`.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **ROUGE-1** | `0.4215` | Unigram overlap between generated and reference summaries |
| **ROUGE-2** | `0.1942` | Bigram overlap measuring fluency and phrase accuracy |
| **ROUGE-L** | `0.3368` | Longest Common Subsequence ($\text{LCS}$) capturing sentence-level structure |
| **ROUGE-Lsum** | `0.3371` | Summary-level $\text{LCS}$ metric across full dialogue turns |

### Sample Inference

**Input Dialogue:**
> **Hannah:** Hey, do you have Betty's number?  
> **Amanda:** Lemme check  
> **Amanda:** Sorry, can't find it.  
> **Amanda:** Ask Larry  
> **Amanda:** He called her last time we were at the park together  
> **Hannah:** I don't know him well  
> **Amanda:** Don't be shy, he's very nice  
> **Hannah:** If you say so..  
> **Hannah:** I'd rather you texted him  
> **Amanda:** Just text him 🙂  
> **Hannah:** Urgh.. Alright  
> **Hannah:** Bye  
> **Amanda:** Bye bye  

**Generated AI Summary:**
> *Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.*

---

## 🛠️ Tech Stack

- **Deep Learning & NLP:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`
- **Backend & UI:** FastAPI, Uvicorn, Gradio, Pydantic
- **MLOps & Packaging:** PyYAML, `python-box`, `ensure`, Docker, GitHub Actions

---

## 👤 Author

**Trijal Sharma**  
- **GitHub:** [@TrijalSharma](https://github.com/TrijalSharma)
- **LinkedIn:** [Trijal Sharma](https://www.linkedin.com/in/trijal)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
