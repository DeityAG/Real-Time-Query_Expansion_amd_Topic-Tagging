import streamlit as st
import re
from collections import deque
from dataclasses import dataclass
from typing import List

# ── PAGE CONFIG  (must be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Query Expansion & Topic Tagging",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
GROQ_MODEL     = "llama-3.1-8b-instant"
CONTEXT_WINDOW = 20
ENTITY_TYPES   = {"PERSON", "GPE", "ORG", "EVENT", "NORP", "FAC", "LOC"}

INTERRUPT_PATTERNS = [
    r"^\s*(brb|brt|back|ok|okay|k|thanks|thank you|got it|noted|alright|sure|"
    r"wait|hold on|one sec|give me a (min|sec|moment)|be right back|"
    r"i.?m back|coming back|just a min|afk)[\.!\?]?\s*$"
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

SAMPLE_CONVERSATIONS = [
    {
        "label": "🏛️ Politics — India → UK",
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
        "label": "🏏 Sports — Cricket → Football",
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
            ("assistant", "It was co-founded by Sam Altman, Elon Musk, Greg Brockman and others in 2015."),
            ("user",      "what did he do after leaving?"),
            ("assistant", "Elon Musk left OpenAI's board in 2018 and later started xAI."),
            ("user",      "and google's version?"),
            ("assistant", "Google's AI lab is called Google DeepMind, known for Gemini models."),
            ("user",      "how do they compare?"),
        ],
    },
]

QUICK_MSGS = [
    "who is pm of india?",
    "what are his duties?",
    "what about uk?",
    "compare both",
    "brb",
    "back — tell me about cricket",
    "how many centuries does he have?",
    "who founded openai?",
]

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .main .block-container { padding-top: 1.4rem; max-width: 1280px; }

  .bubble-user {
    background: #1e3a5f; color: #e8f0fe;
    border-radius: 18px 18px 4px 18px;
    padding: 9px 15px; margin: 5px 0 2px auto;
    max-width: 76%; width: fit-content; font-size: .92rem; margin-left: auto;
  }
  .bubble-assistant {
    background: #1f2937; color: #d1d5db;
    border-radius: 18px 18px 18px 4px;
    padding: 9px 15px; margin: 2px auto 5px 0;
    max-width: 76%; width: fit-content; font-size: .92rem;
  }

  .card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 12px 16px; margin: 7px 0;
  }
  .card .lbl {
    font-size: .68rem; text-transform: uppercase; letter-spacing: .08em;
    color: #6b7280; margin-bottom: 4px;
  }
  .card .val { font-size: .93rem; color: #f3f4f6; font-weight: 500; }
  .expanded-text { color: #34d399; font-style: italic; }

  .tag-pill {
    display: inline-block; padding: 3px 13px; border-radius: 999px;
    font-size: .78rem; font-weight: 700; color: #fff; margin: 2px 3px;
  }
  .entity-pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: .73rem; background: #374151; color: #9ca3af; margin: 2px 3px;
  }

  .conf-wrap {
    background: #1f2937; border-radius: 3px; height: 5px;
    margin-top: 3px; width: 100%;
  }
  .conf-bar { height: 5px; border-radius: 3px; }

  .badge-expanded  { background:#065f46; color:#6ee7b7; font-size:.72rem; padding:2px 9px; border-radius:999px; }
  .badge-interrupt { background:#374151; color:#9ca3af; font-size:.72rem; padding:2px 9px; border-radius:999px; }
  .badge-complete  { background:#1e3a5f; color:#93c5fd; font-size:.72rem; padding:2px 9px; border-radius:999px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CORE CLASSES
# ══════════════════════════════════════════════════════════════════

def is_interruption(text: str) -> bool:
    t, words = text.strip().lower(), text.strip().lower().split()
    q_words = {'who','what','where','when','why','how','which','whose','whom'}
    if len(words) <= 3 and not any(w in q_words for w in words):
        return True
    for p in INTERRUPT_PATTERNS:
        if re.match(p, t, re.IGNORECASE):
            return True
    return False


@dataclass
class EntityEntry:
    text: str; label: str; turn_idx: int


class EntityRegister:
    def __init__(self):
        self.entries: List[EntityEntry] = []

    def update(self, text: str, turn_idx: int, nlp):
        for ent in nlp(text).ents:
            if ent.label_ in ENTITY_TYPES:
                if not any(e.text.lower() == ent.text.lower() and e.turn_idx == turn_idx
                           for e in self.entries):
                    self.entries.append(EntityEntry(ent.text, ent.label_, turn_idx))

    def prune(self, min_turn: int):
        self.entries = [e for e in self.entries if e.turn_idx >= min_turn]

    def get_recent(self, n=5) -> List[EntityEntry]:
        return sorted(self.entries, key=lambda e: e.turn_idx, reverse=True)[:n]

    def as_context_string(self) -> str:
        recent = self.get_recent(8)
        return "(none)" if not recent else ", ".join(f"{e.text} [{e.label}]" for e in recent)


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


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING  — only spaCy + DistilBERT, no local LLM
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models(hf_token: str):
    """
    Loads spaCy NER (en_core_web_sm — CPU-safe) and both DistilBERT
    classifiers from HuggingFace. Groq handles the LLM — no local weights.
    Cold start on Streamlit Cloud: ~30-50s.
    """
    import spacy
    from transformers import pipeline as hf_pipeline
    from huggingface_hub import login

    if hf_token:
        login(token=hf_token)

    # spaCy — sm for CPU (trf needs GPU)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    # DistilBERT topic classifiers — ~66 MB each, fast on CPU
    clf_l1 = hf_pipeline("text-classification",
                          model="Adignite/query-topic-l1-classifier",
                          device=-1)
    clf_l2 = hf_pipeline("text-classification",
                          model="Adignite/query-topic-l2-classifier",
                          device=-1)
    return nlp, clf_l1, clf_l2


# ══════════════════════════════════════════════════════════════════
#  GROQ EXPANSION  — exact same prompt as working notebook
# ══════════════════════════════════════════════════════════════════

EXPANSION_SYSTEM = (
    "Rewrite the latest user message into a fully self-contained query using context.\n"
    "- Replace all pronouns and implicit references with explicit names.\n"
    "- Expand incomplete questions, topic shifts, or comparisons into full sentences.\n"
    "- Output ONLY the rewritten query. No quotes, no explanations. "
    "Return as-is if already self-contained."
)


def groq_expand(history: list, current_msg: str, entity_reg: EntityRegister,
                groq_client) -> str:
    history_str  = "\n".join(
        f"{'User' if t['role']=='user' else 'Assistant'}: {t['text']}"
        for t in history
    )
    user_prompt = (
        f"Conversation context (last {len(history)} messages):\n"
        f"{history_str}\n\n"
        f"Named entities in context: {entity_reg.as_context_string()}\n\n"
        f"Latest user message: {current_msg}\n\n"
        f"Rewrite as a self-contained question:"
    )

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": EXPANSION_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    expanded = resp.choices[0].message.content.strip()

    # Clean up — same logic as notebook
    expanded = expanded.split("\n")[0].strip()
    for prefix in ["Rewritten:", "Answer:", "Query:", "Here is", "Here's", "The question is"]:
        if expanded.lower().startswith(prefix.lower()):
            expanded = expanded[len(prefix):].strip()

    return expanded if expanded else current_msg


# ══════════════════════════════════════════════════════════════════
#  FULL TURN PROCESSOR
# ══════════════════════════════════════════════════════════════════

def process_turn(text, history_deque, entity_reg, nlp, clf_l1, clf_l2,
                 groq_client, turn_idx) -> TurnResult:

    entity_reg.update(text, turn_idx, nlp)
    entity_reg.prune(max(0, turn_idx - CONTEXT_WINDOW))
    entities_used = [e.text for e in entity_reg.get_recent(5)]
    interrupt     = is_interruption(text)

    if interrupt:
        expanded = text
        l1, l1s  = "General", 1.0
        l2, l2s  = "General", 1.0
        was_exp  = False
    else:
        expanded = groq_expand(list(history_deque), text, entity_reg, groq_client)
        was_exp  = expanded.lower().strip() != text.lower().strip()
        r1  = clf_l1(expanded)[0]
        r2  = clf_l2(expanded)[0]
        l1, l1s = r1["label"], r1["score"]
        l2, l2s = r2["label"], r2["score"]

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
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
# Fetch keys silently from the backend
groq_key = st.secrets.get("GROQ_API_KEY", None)
hf_token = st.secrets.get("HF_TOKEN", None)

with st.sidebar:
    st.markdown("## ⚙️  Configuration")
    st.success("API Keys loaded securely from backend.")

    st.markdown("---")
    st.markdown("### 🔁  Pipeline")
    st.markdown("""
**1 · spaCy NER** Rolling entity register from last 20 messages

**2 · Interruption check** `brb`, `wait`, `ok` → tag `General`, skip LLM

**3 · Groq · llama-3.1-8b-instant** Rewrites query using history + entity context

**4 · DistilBERT classifier** Tags `topic_l1` and `topic_l2`
""")

    st.markdown("---")
    st.markdown("### 🏷️  Models")
    st.caption("Adignite/query-topic-l1-classifier")
    st.caption("Adignite/query-topic-l2-classifier")
    st.caption("meta-llama/llama-3.1-8b-instant (Groq)")
    st.caption("en_core_web_sm (spaCy)")

    st.markdown("---")
    if "turn_history" in st.session_state and st.session_state.turn_history:
        turns    = st.session_state.turn_history
        expanded = [t for t in turns if t.was_expanded]
        st.markdown("### 📊  Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Turns",     len(turns))
        c2.metric("Expanded",  len(expanded))
        st.progress(len(st.session_state.history_deque) / CONTEXT_WINDOW,
                    text=f"Context window: {len(st.session_state.history_deque)}/{CONTEXT_WINDOW}")


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("# 🔍  Real-Time Query Expansion & Topic Tagging")
st.markdown(
    "Every user message is **expanded** into a self-contained query "
    "and **tagged** with a hierarchical topic — using the last 20 messages as context."
)

# ── GATE: require both keys ───────────────────────────────────────
if not groq_key or not hf_token:
    st.error("❌ **Configuration Error:** API keys are missing. Please configure `GROQ_API_KEY` and `HF_TOKEN` in your Streamlit secrets.")

    st.markdown("---")
    st.markdown("### 💡  How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""**Query Expansion**
> Raw: `"what are his duties?"`
>
> Expanded: `"What are Narendra Modi's duties as Prime Minister of India?"`""")
    with c2:
        st.markdown("""**Topic Tagging**
> `"compare both"`
>
> `Politics › India`
> confidence: 0.94""")
    with c3:
        st.markdown("""**Interruption Handling**
> `"brb"` / `"wait a sec"`
>
> Skips LLM — tagged
> `General › General`""")
    st.stop()


# ── LOAD MODELS ───────────────────────────────────────────────────
if "models_loaded" not in st.session_state:
    with st.spinner("🔄  Loading spaCy NER + DistilBERT classifiers… (~30s first time)"):
        try:
            nlp, clf_l1, clf_l2 = load_models(hf_token)
            from groq import Groq
            groq_client = Groq(api_key=groq_key)

            st.session_state.models_loaded = True
            st.session_state.nlp        = nlp
            st.session_state.clf_l1     = clf_l1
            st.session_state.clf_l2     = clf_l2
            st.session_state.groq_client = groq_client
        except Exception as e:
            st.error(f"❌  Model loading failed: {e}")
            st.stop()

# ── INIT SESSION STATE ────────────────────────────────────────────
for key, default in [
    ("history_deque", deque(maxlen=CONTEXT_WINDOW)),
    ("entity_reg",    EntityRegister()),
    ("turn_idx",      0),
    ("turn_history",  []),
    ("chat_display",  []),
    ("sample_pending", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════
#  SAMPLE CONVERSATIONS
# ══════════════════════════════════════════════════════════════════

st.markdown("### 📎  Sample Conversations")
s_cols = st.columns(len(SAMPLE_CONVERSATIONS))
for i, (col, sample) in enumerate(zip(s_cols, SAMPLE_CONVERSATIONS)):
    with col:
        if st.button(sample["label"], key=f"sample_{i}", use_container_width=True):
            st.session_state.sample_pending = i

if st.session_state.sample_pending is not None:
    idx    = st.session_state.sample_pending
    sample = SAMPLE_CONVERSATIONS[idx]
    st.session_state.sample_pending = None

    # Reset state
    st.session_state.history_deque = deque(maxlen=CONTEXT_WINDOW)
    st.session_state.entity_reg    = EntityRegister()
    st.session_state.turn_idx      = 0
    st.session_state.turn_history  = []
    st.session_state.chat_display  = []

    with st.spinner(f"Running: {sample['label']} …"):
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
                    st.session_state.clf_l1,
                    st.session_state.clf_l2,
                    st.session_state.groq_client,
                    st.session_state.turn_idx,
                )
                st.session_state.turn_idx += 1
                st.session_state.turn_history.append(result)
                st.session_state.chat_display.append(
                    {"role": "user", "text": text, "result": result}
                )
    st.rerun()


st.markdown("---")

# ══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — CHAT (left) + RESULTS (right)
# ══════════════════════════════════════════════════════════════════
def handle_submit():
    current_msg = st.session_state.user_input.strip()
    if current_msg:
        result = process_turn(
            current_msg,
            st.session_state.history_deque,
            st.session_state.entity_reg,
            st.session_state.nlp,
            st.session_state.clf_l1,
            st.session_state.clf_l2,
            st.session_state.groq_client,
            st.session_state.turn_idx,
        )
        st.session_state.turn_idx += 1
        st.session_state.turn_history.append(result)
        st.session_state.chat_display.append(
            {"role": "user", "text": current_msg, "result": result}
        )
        # It is now 100% safe to clear the input here!
        st.session_state.user_input = ""
        
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: CHAT ────────────────────────────────────────────────────
with left_col:
    st.markdown("### 💬  Conversation")

    if st.button("🗑️  Reset Conversation", use_container_width=True):
        st.session_state.history_deque = deque(maxlen=CONTEXT_WINDOW)
        st.session_state.entity_reg    = EntityRegister()
        st.session_state.turn_idx      = 0
        st.session_state.turn_history  = []
        st.session_state.chat_display  = []
        st.rerun()

    # Chat bubbles
    chat_box = st.container(height=420)
    with chat_box:
        if not st.session_state.chat_display:
            st.markdown(
                "<p style='color:#4b5563;text-align:center;margin-top:70px;'>"
                "Load a sample ↑ or type a message below</p>",
                unsafe_allow_html=True,
            )
        for item in st.session_state.chat_display:
            if item["role"] == "user":
                st.markdown(
                    f'<div class="bubble-user">👤 {item["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bubble-assistant">🤖 {item["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # Quick-input buttons
    st.markdown("**Quick inputs:**")
    qcols = st.columns(4)
    for qi, qmsg in enumerate(QUICK_MSGS):
        with qcols[qi % 4]:
            if st.button(qmsg, key=f"q{qi}", use_container_width=True):
                # Fixed: update session_state matching text_input key directly
                st.session_state.user_input = qmsg

    # Fixed: removed value=... and using key only
    user_input = st.text_input(
        "Message",
        placeholder="e.g. 'what are his duties?' / 'what about uk?' / 'brb'",
        label_visibility="collapsed",
        key="user_input",
        on_change=handle_submit
    )

    # Optional: add bot response to context
    with st.expander("➕  Add bot response to context"):
        bot_text = st.text_input(
            "Bot reply",
            placeholder="e.g. 'Narendra Modi has been PM since 2014.'",
            key="bot_input",
        )
        if st.button("Add to context", use_container_width=True):
            if bot_text.strip():
                st.session_state.history_deque.append(
                    {"role": "assistant", "text": bot_text.strip()}
                )
                st.session_state.entity_reg.update(
                    bot_text.strip(), st.session_state.turn_idx, st.session_state.nlp
                )
                st.session_state.turn_idx += 1
                st.session_state.chat_display.append(
                    {"role": "assistant", "text": bot_text.strip()}
                )
                st.rerun()

    st.button("🚀  Send", type="primary", use_container_width=True, on_click=handle_submit)


# ── RIGHT: RESULTS ────────────────────────────────────────────────
with right_col:
    st.markdown("### 🧠  Analysis Results")

    if not st.session_state.turn_history:
        st.markdown(
            "<p style='color:#4b5563;margin-top:80px;text-align:center;'>"
            "Results appear here after each message.</p>",
            unsafe_allow_html=True,
        )
    else:
        latest = st.session_state.turn_history[-1]
        color  = TOPIC_COLORS.get(latest.topic_l1, "#9CA3AF")

        st.markdown("#### 🔎  Latest Turn")

        if latest.is_interruption:
            badge = '<span class="badge-interrupt">⏸ Interruption</span>'
        elif latest.was_expanded:
            badge = '<span class="badge-expanded">✦ Expanded</span>'
        else:
            badge = '<span class="badge-complete">✓ Already complete</span>'

        st.markdown(f"""
<div class="card">
  <div class="lbl">RAW MESSAGE</div>
  <div class="val">{latest.raw_message}</div>
</div>
<div class="card">
  <div class="lbl">EXPANDED QUERY &nbsp; {badge}</div>
  <div class="val expanded-text">{latest.expanded_query}</div>
</div>
""", unsafe_allow_html=True)

        l1b = int(latest.topic_l1_score * 100)
        l2b = int(latest.topic_l2_score * 100)
        st.markdown(f"""
<div class="card">
  <div class="lbl">TOPIC</div>
  <span class="tag-pill" style="background:{color};">{latest.topic_l1}</span>
  <span style="color:#6b7280;font-size:1.1rem;"> › </span>
  <span class="tag-pill" style="background:{color}99;">{latest.topic_l2}</span>
  <br><br>
  <div style="display:flex;gap:18px;">
    <div style="flex:1;">
      <div style="font-size:.7rem;color:#6b7280;">L1 confidence</div>
      <div class="conf-wrap">
        <div class="conf-bar" style="width:{l1b}%;background:{color};"></div>
      </div>
      <div style="font-size:.75rem;color:#9ca3af;margin-top:2px;">{latest.topic_l1_score:.3f}</div>
    </div>
    <div style="flex:1;">
      <div style="font-size:.7rem;color:#6b7280;">L2 confidence</div>
      <div class="conf-wrap">
        <div class="conf-bar" style="width:{l2b}%;background:{color}99;"></div>
      </div>
      <div style="font-size:.75rem;color:#9ca3af;margin-top:2px;">{latest.topic_l2_score:.3f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        if latest.entities_used:
            pills = "".join(
                f'<span class="entity-pill">{e}</span>' for e in latest.entities_used
            )
            st.markdown(f"""
<div class="card">
  <div class="lbl">ENTITIES IN CONTEXT (spaCy NER)</div>
  <div style="margin-top:5px;">{pills}</div>
</div>
""", unsafe_allow_html=True)

        if len(st.session_state.turn_history) > 1:
            st.markdown("#### 📜  Turn History")
            import pandas as pd
            rows = []
            for t in reversed(st.session_state.turn_history):
                rows.append({
                    "Raw":      t.raw_message[:38] + ("…" if len(t.raw_message) > 38 else ""),
                    "Expanded": t.expanded_query[:50] + ("…" if len(t.expanded_query) > 50 else ""),
                    "L1":       t.topic_l1,
                    "L2":       t.topic_l2,
                    "Conf":     f"{t.topic_l1_score:.2f}",
                    "⤴":        "✓" if t.was_expanded else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
