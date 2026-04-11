import streamlit as st
import os
import re
import json
import torch
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Query Expansion",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SAMPLE QUESTIONS ──────────────────────────────────────────────
SAMPLE_CONVERSATIONS = [
    {
        "label": "🏛️ Politics — India → UK switch",
        "turns": [
            ("user",      "who is pm of india?"),
            ("assistant", "The Prime Minister of India is Narendra Modi, in office since 2014."),
            ("user",      "what are his duties?"),
            ("assistant", "He leads the Union Cabinet, sets national policy, and represents India globally."),
            ("user",      "and internationally?"),
            ("assistant", "He attends G20, UN summits and handles bilateral diplomacy."),
            ("user",      "wait a sec"),
            ("assistant", "Sure, take your time!"),
            ("user",      "back — what about uk?"),
            ("assistant", "The Prime Minister of the UK is Rishi Sunak."),
            ("user",      "compare both"),
        ],
    },
    {
        "label": "🏏 Sports — Cricket then Football",
        "turns": [
            ("user",      "who is the best cricket player?"),
            ("assistant", "Virat Kohli is widely considered one of the greatest batsmen today."),
            ("user",      "how many centuries does he have?"),
            ("assistant", "Kohli has over 80 international centuries across all formats."),
            ("user",      "what about ms dhoni?"),
            ("assistant", "MS Dhoni is legendary for his captaincy and led India to the 2011 World Cup."),
            ("user",      "compare both their batting"),
            ("assistant", "Kohli is aggressive; Dhoni is famous for his calm finishing."),
            ("user",      "brb"),
            ("assistant", "Take your time!"),
            ("user",      "ok back. now what about messi?"),
        ],
    },
    {
        "label": "🤖 Technology — AI",
        "turns": [
            ("user",      "tell me about openai"),
            ("assistant", "OpenAI is an AI research company known for ChatGPT and GPT-4."),
            ("user",      "who founded it?"),
            ("assistant", "It was founded by Sam Altman, Elon Musk, Greg Brockman and others in 2015."),
            ("user",      "what did he do after leaving?"),
            ("assistant", "Elon Musk departed OpenAI's board in 2018 and later started xAI."),
            ("user",      "and google's version?"),
        ],
    },
]

TOPIC_COLORS = {
    "Politics":      "#3B82F6",
    "Sports":        "#10B981",
    "Technology":    "#8B5CF6",
    "Entertainment": "#F59E0B",
    "Health":        "#EF4444",
    "History":       "#6B7280",
    "Geography":     "#14B8A6",
    "General":       "#9CA3AF",
}

# ── CONSTANTS (must match notebook exactly) ───────────────────────
CONTEXT_WINDOW    = 20
MAX_NEW_TOKENS    = 200
ENTITY_TYPES      = {"PERSON", "GPE", "ORG", "EVENT", "NORP", "FAC", "LOC"}
INTERRUPT_PATTERNS = [
    r'^\s*(brb|brt|back|ok|okay|k|thanks|thank you|got it|noted|alright|sure|'
    r'wait|hold on|one sec|give me a (min|sec|moment)|be right back|'
    r'i.?m back|coming back|just a min|afk)[\.!\?]?\s*$'
]

# ── INTERRUPTION DETECTOR ─────────────────────────────────────────
def is_interruption(text: str) -> bool:
    t = text.strip().lower()
    words = t.split()
    question_words = {'who','what','where','when','why','how','which','whose','whom'}
    if len(words) <= 3 and not any(w in question_words for w in words):
        return True
    for pattern in INTERRUPT_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    return False

# ── ENTITY REGISTER ───────────────────────────────────────────────
@dataclass
class EntityEntry:
    text:     str
    label:    str
    turn_idx: int

class EntityRegister:
    def __init__(self):
        self.entries: List[EntityEntry] = []

    def update(self, text: str, turn_idx: int, nlp):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                if not any(e.text.lower() == ent.text.lower() and e.turn_idx == turn_idx
                           for e in self.entries):
                    self.entries.append(EntityEntry(text=ent.text, label=ent.label_, turn_idx=turn_idx))

    def prune(self, min_turn: int):
        self.entries = [e for e in self.entries if e.turn_idx >= min_turn]

    def get_recent(self, n: int = 5) -> List[EntityEntry]:
        return sorted(self.entries, key=lambda e: e.turn_idx, reverse=True)[:n]

    def as_context_string(self) -> str:
        recent = self.get_recent(8)
        if not recent:
            return "(none)"
        return ", ".join(f"{e.text} [{e.label}]" for e in recent)

# ── TURN RESULT ───────────────────────────────────────────────────
@dataclass
class TurnResult:
    raw_message:     str
    expanded_query:  str
    topic_l1:        str
    topic_l1_score:  float
    topic_l2:        str
    topic_l2_score:  float
    entities_used:   List[str]
    was_expanded:    bool
    is_interruption: bool

# ── MODEL LOADING (cached) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(hf_token: str):
    import spacy
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    from huggingface_hub import login

    login(token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # spaCy NER
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        nlp = spacy.load("en_core_web_sm")

    # LLM
    llm_id = "meta-llama/Llama-3.2-1B-Instruct"
    llm_tok = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        llm_model = llm_model.to(device)
    llm_model.eval()

    has_chat = hasattr(llm_tok, 'apply_chat_template') and llm_tok.chat_template is not None

    # Classifiers
    clf_l1 = hf_pipeline('text-classification',
                          model="Adignite/query-topic-l1-classifier",
                          device=0 if device == "cuda" else -1)
    clf_l2 = hf_pipeline('text-classification',
                          model="Adignite/query-topic-l2-classifier",
                          device=0 if device == "cuda" else -1)

    return nlp, llm_tok, llm_model, has_chat, clf_l1, clf_l2, device


# ── LLM EXPANSION (exact logic from notebook) ─────────────────────
EXPANSION_SYSTEM = (
    "Rewrite the latest user message into a fully self-contained query using context.\n"
    "- Replace all pronouns and implicit references with explicit names.\n"
    "- Expand incomplete questions, topic shifts, or comparisons into full sentences.\n"
    "- Output ONLY the rewritten query. No quotes, no explanations. Return as-is if already self-contained."
)

def llm_expand(history, current_msg, entity_register, llm_tok, llm_model, has_chat, device):
    history_str  = "\n".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['text']}"
        for t in history
    )
    entities_str = entity_register.as_context_string()
    user_prompt  = (
        f"Conversation context (last {len(history)} messages):\n"
        f"{history_str}\n\n"
        f"Named entities in context: {entities_str}\n\n"
        f"Latest user message: {current_msg}\n\n"
        f"Rewrite as a self-contained question:"
    )

    if has_chat:
        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        encoded   = llm_tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        encoded   = {k: v.to(device) for k, v in encoded.items()}
        input_ids = encoded["input_ids"]
    else:
        combined  = f"### System:\n{EXPANSION_SYSTEM}\n\n### User:\n{user_prompt}\n\n### Response:\n"
        encoded   = llm_tok(combined, return_tensors="pt")
        encoded   = {k: v.to(device) for k, v in encoded.items()}
        input_ids = encoded["input_ids"]

    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = llm_model.generate(
            input_ids,
            attention_mask = encoded.get("attention_mask"),
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.3,
            do_sample      = True,
            pad_token_id   = llm_tok.pad_token_id,
            eos_token_id   = llm_tok.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    expanded   = llm_tok.decode(new_tokens, skip_special_tokens=True).strip()
    expanded   = expanded.split('\n')[0].strip()
    for prefix in ["Rewritten:", "Answer:", "Query:", "Here is", "Here's", "The question is"]:
        if expanded.lower().startswith(prefix.lower()):
            expanded = expanded[len(prefix):].strip()

    return expanded if expanded else current_msg


# ── PROCESS ONE USER TURN ─────────────────────────────────────────
def process_turn(text, history_deque, entity_reg, nlp, llm_tok, llm_model,
                 has_chat, clf_l1, clf_l2, device, turn_idx):

    entity_reg.update(text, turn_idx, nlp)
    min_turn = max(0, turn_idx - CONTEXT_WINDOW)
    entity_reg.prune(min_turn)

    entities_used = [e.text for e in entity_reg.get_recent(5)]
    interrupt     = is_interruption(text)

    if interrupt:
        expanded = text
        l1, l1s  = "General", 1.0
        l2, l2s  = "General", 1.0
        was_exp  = False
    else:
        history_list = list(history_deque)
        expanded = llm_expand(history_list, text, entity_reg, llm_tok, llm_model, has_chat, device)
        was_exp  = expanded.lower().strip() != text.lower().strip()
        r1  = clf_l1(expanded)[0]
        r2  = clf_l2(expanded)[0]
        l1, l1s = r1['label'], r1['score']
        l2, l2s = r2['label'], r2['score']

    history_deque.append({"role": "user", "text": text})

    return TurnResult(
        raw_message=text, expanded_query=expanded,
        topic_l1=l1, topic_l1_score=l1s,
        topic_l2=l2, topic_l2_score=l2s,
        entities_used=entities_used,
        was_expanded=was_exp,
        is_interruption=interrupt,
    )


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  /* overall background */
  .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

  /* chat bubble user */
  .bubble-user {
    background: #1e3a5f;
    color: #e8f0fe;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 6px 0 2px auto;
    max-width: 72%;
    width: fit-content;
    font-size: 0.95rem;
    margin-left: auto;
  }
  /* chat bubble assistant */
  .bubble-assistant {
    background: #1f2937;
    color: #d1d5db;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 2px auto 6px 0;
    max-width: 72%;
    width: fit-content;
    font-size: 0.95rem;
  }
  /* result card */
  .result-card {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .result-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 3px;
  }
  .result-card .value {
    font-size: 0.95rem;
    color: #f3f4f6;
    font-weight: 500;
  }
  .expanded-text {
    color: #34d399;
    font-style: italic;
  }
  .tag-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
    margin: 2px 3px;
  }
  .entity-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: #374151;
    color: #9ca3af;
    margin: 2px 3px;
  }
  .confidence-bar-wrap {
    background: #1f2937;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
    width: 100%;
  }
  .confidence-bar {
    height: 6px;
    border-radius: 4px;
  }
  /* sample pill button area */
  .sample-btn {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #9ca3af;
    cursor: pointer;
    display: inline-block;
    margin: 3px;
  }
  .processing-badge {
    background: #065f46;
    color: #6ee7b7;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 999px;
    display: inline-block;
  }
  .interrupt-badge {
    background: #374151;
    color: #9ca3af;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 999px;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        help="Required for Llama-3.2-1B-Instruct and classifier models",
        placeholder="hf_..."
    )

    st.markdown("---")
    st.markdown("### 📋 Pipeline")
    st.markdown("""
**Step 1 — spaCy NER**  
Extracts entities from rolling 20-message window  

**Step 2 — Interruption Check**  
Small talk → tag `General`, skip LLM  

**Step 3 — Llama-3.2-1B-Instruct**  
Rewrites query using history + entities  

**Step 4 — DistilBERT Classifier**  
Tags `topic_l1` and `topic_l2`
""")
    st.markdown("---")
    st.markdown("### 🏷️ Models")
    st.markdown("""
- `Adignite/query-topic-l1-classifier`
- `Adignite/query-topic-l2-classifier`
- `meta-llama/Llama-3.2-1B-Instruct`
- `en_core_web_trf` (spaCy)
""")
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if "turn_history" in st.session_state:
        turns = st.session_state.turn_history
        expanded = [t for t in turns if t.was_expanded]
        st.metric("Total turns", len(turns))
        st.metric("Expansions performed", len(expanded))
        st.metric("Context window used",
                  f"{min(len(st.session_state.history_deque), CONTEXT_WINDOW)}/{CONTEXT_WINDOW}")


# ── MAIN AREA ─────────────────────────────────────────────────────
st.markdown("# 🔍 Real-Time Query Expansion & Topic Tagging")
st.markdown("*Type a message — the system expands implicit queries and tags them with a topic in real time.*")

# ── MODEL LOAD ────────────────────────────────────────────────────
if not hf_token:
    st.info("👈 Enter your HuggingFace token in the sidebar to load models and begin.")
    st.markdown("---")
    st.markdown("### 💡 What this system does")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Query Expansion**
> `"what are his duties?"`
> ↓
> `"What are Narendra Modi's duties as Prime Minister of India?"`
""")
    with col2:
        st.markdown("""
**Topic Tagging**
> `"compare both"`
> ↓
> `Politics > India`
> confidence: 0.94
""")
    with col3:
        st.markdown("""
**Interruption Handling**
> `"brb"` / `"wait a sec"`
> ↓
> Skips LLM, tagged
> `General > General`
""")
    st.stop()


# ── INIT SESSION STATE ────────────────────────────────────────────
if "models_loaded" not in st.session_state:
    with st.spinner("🔄 Loading models (this takes ~60s the first time)…"):
        try:
            (nlp, llm_tok, llm_model, has_chat,
             clf_l1, clf_l2, device) = load_models(hf_token)
            st.session_state.models_loaded = True
            st.session_state.nlp       = nlp
            st.session_state.llm_tok   = llm_tok
            st.session_state.llm_model = llm_model
            st.session_state.has_chat  = has_chat
            st.session_state.clf_l1    = clf_l1
            st.session_state.clf_l2    = clf_l2
            st.session_state.device    = device
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            st.stop()

if "history_deque" not in st.session_state:
    st.session_state.history_deque = deque(maxlen=CONTEXT_WINDOW)
if "entity_reg" not in st.session_state:
    st.session_state.entity_reg = EntityRegister()
if "turn_idx" not in st.session_state:
    st.session_state.turn_idx = 0
if "turn_history" not in st.session_state:
    st.session_state.turn_history = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []   # [{role, text, result?}]
if "sample_pending" not in st.session_state:
    st.session_state.sample_pending = None


# ── SAMPLE CONVERSATIONS ──────────────────────────────────────────
st.markdown("### 📎 Load a Sample Conversation")
sample_cols = st.columns(len(SAMPLE_CONVERSATIONS))
for i, (col, sample) in enumerate(zip(sample_cols, SAMPLE_CONVERSATIONS)):
    with col:
        if st.button(sample["label"], key=f"sample_{i}", use_container_width=True):
            st.session_state.sample_pending = i

# Run sample if one was selected
if st.session_state.sample_pending is not None:
    idx    = st.session_state.sample_pending
    sample = SAMPLE_CONVERSATIONS[idx]
    st.session_state.sample_pending = None

    # Reset conversation
    st.session_state.history_deque = deque(maxlen=CONTEXT_WINDOW)
    st.session_state.entity_reg    = EntityRegister()
    st.session_state.turn_idx      = 0
    st.session_state.turn_history  = []
    st.session_state.chat_display  = []

    with st.spinner(f"Running sample: {sample['label']} …"):
        for role, text in sample["turns"]:
            if role == "assistant":
                st.session_state.history_deque.append({"role": "assistant", "text": text})
                st.session_state.entity_reg.update(
                    text, st.session_state.turn_idx, st.session_state.nlp
                )
                st.session_state.turn_idx += 1
                st.session_state.chat_display.append({"role": "assistant", "text": text})
            else:
                result = process_turn(
                    text,
                    st.session_state.history_deque,
                    st.session_state.entity_reg,
                    st.session_state.nlp,
                    st.session_state.llm_tok,
                    st.session_state.llm_model,
                    st.session_state.has_chat,
                    st.session_state.clf_l1,
                    st.session_state.clf_l2,
                    st.session_state.device,
                    st.session_state.turn_idx,
                )
                st.session_state.turn_idx += 1
                st.session_state.turn_history.append(result)
                st.session_state.chat_display.append({"role": "user", "text": text, "result": result})
    st.rerun()


st.markdown("---")

# ── CHAT + RESULTS LAYOUT ─────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 💬 Conversation")

    # Reset button
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.history_deque = deque(maxlen=CONTEXT_WINDOW)
        st.session_state.entity_reg    = EntityRegister()
        st.session_state.turn_idx      = 0
        st.session_state.turn_history  = []
        st.session_state.chat_display  = []
        st.rerun()

    # Chat display
    chat_container = st.container(height=440)
    with chat_container:
        if not st.session_state.chat_display:
            st.markdown(
                "<p style='color:#4b5563;text-align:center;margin-top:80px;'>"
                "Start typing or load a sample conversation ↑</p>",
                unsafe_allow_html=True
            )
        for item in st.session_state.chat_display:
            if item["role"] == "user":
                st.markdown(
                    f'<div class="bubble-user">👤 {item["text"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bubble-assistant">🤖 {item["text"]}</div>',
                    unsafe_allow_html=True
                )

    # Message input
    st.markdown("#### ✏️ Your message")

    # Quick sample input suggestions
    quick_msgs = [
        "who is pm of india?",
        "what are his duties?",
        "what about uk?",
        "compare both",
        "brb",
        "back — tell me about cricket",
        "how many centuries does he have?",
        "who founded openai?",
    ]
    st.markdown("**Quick inputs:**")
    qcols = st.columns(4)
    for qi, qmsg in enumerate(quick_msgs):
        with qcols[qi % 4]:
            if st.button(qmsg, key=f"quick_{qi}", use_container_width=True):
                st.session_state["prefill_msg"] = qmsg

    user_input = st.text_input(
        "Type your message",
        value=st.session_state.pop("prefill_msg", ""),
        placeholder="e.g. 'what are his duties?' or 'what about uk?'",
        label_visibility="collapsed",
        key="user_msg_input",
    )

    # Bot response input (for adding assistant context)
    with st.expander("➕ Add bot response to context (optional)"):
        bot_input = st.text_input(
            "Bot reply",
            placeholder="e.g. 'Narendra Modi has been PM since 2014.'",
            key="bot_msg_input",
        )
        if st.button("Add bot response", use_container_width=True):
            if bot_input.strip():
                st.session_state.history_deque.append({"role": "assistant", "text": bot_input.strip()})
                st.session_state.entity_reg.update(
                    bot_input.strip(), st.session_state.turn_idx, st.session_state.nlp
                )
                st.session_state.turn_idx += 1
                st.session_state.chat_display.append({"role": "assistant", "text": bot_input.strip()})
                st.rerun()

    if st.button("🚀 Send", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing…"):
                result = process_turn(
                    user_input.strip(),
                    st.session_state.history_deque,
                    st.session_state.entity_reg,
                    st.session_state.nlp,
                    st.session_state.llm_tok,
                    st.session_state.llm_model,
                    st.session_state.has_chat,
                    st.session_state.clf_l1,
                    st.session_state.clf_l2,
                    st.session_state.device,
                    st.session_state.turn_idx,
                )
                st.session_state.turn_idx += 1
                st.session_state.turn_history.append(result)
                st.session_state.chat_display.append({
                    "role": "user", "text": user_input.strip(), "result": result
                })
            st.rerun()


# ── RIGHT PANEL: RESULTS ──────────────────────────────────────────
with right_col:
    st.markdown("### 🧠 Analysis Results")

    if not st.session_state.turn_history:
        st.markdown(
            "<p style='color:#4b5563;margin-top:80px;text-align:center;'>"
            "Results will appear here as you send messages.</p>",
            unsafe_allow_html=True
        )
    else:
        # Show latest result prominently
        latest = st.session_state.turn_history[-1]
        color  = TOPIC_COLORS.get(latest.topic_l1, "#9CA3AF")

        st.markdown("#### 🔎 Latest Turn")

        # Expanded query card
        badge = '<span class="interrupt-badge">⏸ Interruption</span>' if latest.is_interruption \
                else ('<span class="processing-badge">✦ Expanded</span>' if latest.was_expanded
                      else '<span style="color:#6b7280;font-size:0.75rem;">Already complete</span>')

        st.markdown(f"""
<div class="result-card">
  <div class="label">RAW MESSAGE</div>
  <div class="value">{latest.raw_message}</div>
</div>
<div class="result-card">
  <div class="label">EXPANDED QUERY &nbsp; {badge}</div>
  <div class="value expanded-text">{latest.expanded_query}</div>
</div>
""", unsafe_allow_html=True)

        # Topic pills
        l1_bar = int(latest.topic_l1_score * 100)
        l2_bar = int(latest.topic_l2_score * 100)
        st.markdown(f"""
<div class="result-card">
  <div class="label">TOPIC CLASSIFICATION</div>
  <span class="tag-pill" style="background:{color};">{latest.topic_l1}</span>
  <span style="color:#6b7280;font-size:1.1rem;">›</span>
  <span class="tag-pill" style="background:{color}88;">{latest.topic_l2}</span>
  <br><br>
  <div style="display:flex;gap:16px;">
    <div style="flex:1;">
      <div style="font-size:0.72rem;color:#6b7280;">L1 confidence</div>
      <div class="confidence-bar-wrap">
        <div class="confidence-bar" style="width:{l1_bar}%;background:{color};"></div>
      </div>
      <div style="font-size:0.78rem;color:#9ca3af;margin-top:2px;">{latest.topic_l1_score:.3f}</div>
    </div>
    <div style="flex:1;">
      <div style="font-size:0.72rem;color:#6b7280;">L2 confidence</div>
      <div class="confidence-bar-wrap">
        <div class="confidence-bar" style="width:{l2_bar}%;background:{color}88;"></div>
      </div>
      <div style="font-size:0.78rem;color:#9ca3af;margin-top:2px;">{latest.topic_l2_score:.3f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Entities
        if latest.entities_used:
            pills = "".join(f'<span class="entity-pill">{e}</span>' for e in latest.entities_used)
            st.markdown(f"""
<div class="result-card">
  <div class="label">ENTITIES IN CONTEXT (spaCy NER)</div>
  <div style="margin-top:4px;">{pills}</div>
</div>
""", unsafe_allow_html=True)

        # History table
        if len(st.session_state.turn_history) > 1:
            st.markdown("#### 📜 Turn History")
            rows = []
            for t in reversed(st.session_state.turn_history):
                rows.append({
                    "Raw": t.raw_message[:40] + ("…" if len(t.raw_message)>40 else ""),
                    "Expanded": t.expanded_query[:50] + ("…" if len(t.expanded_query)>50 else ""),
                    "L1": t.topic_l1,
                    "L2": t.topic_l2,
                    "Conf": f"{t.topic_l1_score:.2f}",
                    "⤴": "✓" if t.was_expanded else "—",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
