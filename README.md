# 🧠 DataAnalytica — Multi-Agent Autonomous System for Intelligent Data Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-8A2BE2?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> Upload a CSV. Get a full data science report — automatically.

DataAnalytica is a **7-agent autonomous analytics platform** that automates the complete data science lifecycle from a single CSV upload — profiling, preprocessing, ML model training, diagnostics, visualization, and LLM-powered insight generation — with no code required.

---

## 🚀 Live Demo

> 🔗 **[Coming Soon — Streamlit Cloud](#)**  
> *(Deploy link will be added here)*

---

## 📸 Screenshots

> *(Add screenshots of your app UI here after deployment)*  
> Suggested: Dashboard view, Model results table, LLM Q&A interface

---

## 🤖 The 7-Agent Architecture

Each agent handles one stage of the data science pipeline autonomously:

| Agent | Responsibility |
|---|---|
| **1. Profiling Agent** | Dataset overview — shape, types, nulls, duplicates |
| **2. Preprocessing Agent** | Median/mode imputation, encoding, StandardScaler normalization |
| **3. Visualization Agent** | Correlation heatmaps, distributions, box plots, scatter charts |
| **4. ML Training Agent** | Auto-detects classification vs regression; trains 15 models |
| **5. Diagnostic Agent** | Flags overfitting, class imbalance, performance tiers |
| **6. Insight Agent** | Groq LLaMA 3.3 70B generates JSON-formatted categorized findings |
| **7. Q&A Agent** | Natural language interface — ask questions about your dataset |

---

## ⚙️ Features

- **Zero-code analytics** — upload CSV, get full report instantly
- **Auto task detection** — classifies problem as classification or regression using target-column heuristics
- **15 ML models trained** — across Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, XGBoost, and more
- **5-fold cross-validation** with 80/20 stratified split — prevents data leakage
- **Interactive visualizations** — built with Plotly
- **LLM-powered insights** — Groq LLaMA 3.3 70B explains results in plain English
- **Natural language Q&A** — ask "Which feature has the most impact?" and get an answer
- **Session-state caching** via content-hash keys — reduces multi-hour workflows to under 30 seconds

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit |
| **ML & Modeling** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Visualization** | Plotly |
| **LLM Integration** | Groq API — LLaMA 3.3 70B |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
DataAnalytica/
├── app.py                  # Main Streamlit entry point
├── agents/
│   ├── profiling_agent.py
│   ├── preprocessing_agent.py
│   ├── visualization_agent.py
│   ├── training_agent.py
│   ├── diagnostic_agent.py
│   ├── insight_agent.py
│   └── qa_agent.py
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```

---

## 🔧 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/codewith-jaimin/DataAnalytica.git
cd DataAnalytica
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your Groq API key
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 How It Works

```
Upload CSV
    ↓
Profiling Agent     →  Shape, nulls, data types
    ↓
Preprocessing Agent →  Imputation, encoding, scaling
    ↓
Visualization Agent →  Heatmaps, distributions, plots
    ↓
ML Training Agent   →  15 models, CV, best model selection
    ↓
Diagnostic Agent    →  Overfitting, imbalance, performance flags
    ↓
Insight Agent       →  LLM-generated findings (JSON structured)
    ↓
Q&A Agent           →  Ask anything about your data
```

---

## 📈 Sample Results

> *(Add a table or screenshot of model comparison results here)*  
> Example:

| Model | Accuracy | F1 Score | CV Score |
|---|---|---|---|
| Random Forest | 94.2% | 0.93 | 0.91 |
| Gradient Boosting | 93.8% | 0.92 | 0.90 |
| XGBoost | 92.1% | 0.91 | 0.89 |

---

## 🧩 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
groq
python-dotenv
```

---

## 👤 Author

**Jaimin Pancholi**  
M.Sc. Big Data Analytics — St. Xavier's College, Ahmedabad  
📧 jaiminpancholi5@gmail.com  
🔗 [linkedin.com/in/jaimin-pancholi](https://linkedin.com/in/jaimin-pancholi-266734270)  
🐙 [github.com/codewith-jaimin](https://github.com/codewith-jaimin)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project useful, consider giving it a star — it helps others discover it!
