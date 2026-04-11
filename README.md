# 🔍 Real-Time Query Expansion & Topic Tagging

End-to-end Streamlit app that watches a live conversation and for every user message:
- **Expands** implicit queries (pronouns, ellipsis, topic switches) into self-contained questions
- **Tags** each message with a 2-level topic label (e.g. `Politics > India`)

## Models Used
| Component | Model |
|-----------|-------|
| Topic L1 Classifier | `Adignite/query-topic-l1-classifier` |
| Topic L2 Classifier | `Adignite/query-topic-l2-classifier` |
| Query Expansion LLM | `meta-llama/Llama-3.2-1B-Instruct` |
| Named Entity Recognition | `en_core_web_trf` (spaCy) |

---

## 🖥️ Run Locally

### 1. Clone / copy these files
```
streamlit_app/
├── app.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml        ← add your HF token here (not committed to git)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `en_core_web_trf` (spaCy transformer NER) requires a one-time download.  
> If it fails via requirements.txt, run manually:
> ```bash
> python -m spacy download en_core_web_trf
> ```

### 3. Set your HuggingFace token
Edit `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_token_here"
```
Or just paste it in the sidebar when the app starts.

### 4. Run
```bash
streamlit run app.py
```
Opens at http://localhost:8501

---

## ☁️ Deploy to Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add app.py requirements.txt .streamlit/config.toml .gitignore
# DO NOT add secrets.toml
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/query-expansion-app.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   HF_TOKEN = "hf_your_token_here"
   ```
5. Click **Deploy**

> ⚠️ First load takes ~3–5 minutes as models are downloaded and cached.  
> Subsequent loads are instant (Streamlit caches with `@st.cache_resource`).

---

## 🔧 GPU / Performance Notes

| Environment | Expected speed per turn |
|-------------|------------------------|
| CPU only    | ~15–30s (Llama inference is slow) |
| T4 GPU      | ~3–5s |
| A10/A100    | <2s |

For **CPU-only deployment** (Streamlit Cloud free tier), consider switching the LLM to:
```python
LLM_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"   # much smaller, faster on CPU
```
And swap NER to:
```bash
python -m spacy download en_core_web_sm
```
Then in app.py change `en_core_web_trf` → `en_core_web_sm`.

---

## 📁 File Structure
```
app.py               Main Streamlit application
requirements.txt     Python dependencies
.streamlit/
  config.toml        Dark theme + server config
  secrets.toml       HF token (local only, not committed)
.gitignore           Excludes secrets.toml
README.md            This file
```
