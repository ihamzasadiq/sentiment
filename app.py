import logging
import os

import streamlit as st
from transformers import Pipeline, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.getLogger("transformers.utils.loading_report").setLevel(logging.CRITICAL)

METADATA_VALUE = "huggingface_AI_model"
TOPICS = ["Food Quality", "Service", "Price", "Location", "Ambience"]
TOPIC_TEMPLATE = "The main topic of this review is {}."
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
SENTIMENT_TEMPLATE = "The sentiment of this review is {}."

# --------------------------------------------------------------------------
# Model selection.
#
# APP_MODE=lite (default, Streamlit Cloud friendly ~500 MB of models):
#   one shared zero-shot NLI model (distilbert-mnli, 67M) does sentiment AND
#   topics, plus t5-small (60M) for summaries.
# APP_MODE=full (best accuracy, as evaluated in sentiment.ipynb):
#   BERTweet for sentiment, mDeBERTa zero-shot for topics, distilbart for
#   summaries (~2.5 GB of models — needs a machine with enough RAM).
# Every model can also be overridden individually via env vars.
# --------------------------------------------------------------------------
APP_MODE = os.getenv("APP_MODE", "lite")

if APP_MODE == "full":
    SENTIMENT_MODEL = os.getenv(
        "SENTIMENT_MODEL", "finiteautomata/bertweet-base-sentiment-analysis")
    ZS_MODEL = os.getenv("ZS_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    SUM_MODEL = os.getenv("SUM_MODEL", "sshleifer/distilbart-cnn-12-6")
    SENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
else:
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "typeform/distilbert-base-uncased-mnli")
    ZS_MODEL = os.getenv("ZS_MODEL", "typeform/distilbert-base-uncased-mnli")
    SUM_MODEL = os.getenv("SUM_MODEL", "t5-small")
    SENT_MAP = None


class CustomPipeline(Pipeline):
    """HF pipeline whose outputs all carry {'metadata': 'huggingface_AI_model'}."""

    def __init__(self, task, model=None, **kwargs):
        self._pipe = pipeline(task, model=model, **kwargs)
        super().__init__(model=self._pipe.model, tokenizer=self._pipe.tokenizer,
                         task=task, device=self._pipe.device)

    def __call__(self, inputs, *args, **kwargs):
        outputs = self._pipe(inputs, *args, **kwargs)
        if isinstance(outputs, dict):
            outputs["metadata"] = METADATA_VALUE
        else:
            for o in outputs:
                if isinstance(o, dict):
                    o["metadata"] = METADATA_VALUE
        return outputs

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs, **kwargs):
        raise NotImplementedError

    def _forward(self, model_inputs, **kwargs):
        raise NotImplementedError

    def postprocess(self, model_outputs, **kwargs):
        raise NotImplementedError


@st.cache_resource(show_spinner="Loading zero-shot NLI model ...")
def get_nli_pipeline(model_id):
    return CustomPipeline("zero-shot-classification", model=model_id, device=-1)


@st.cache_resource(show_spinner="Loading sentiment model ...")
def get_sentiment_pipeline(model_id):
    return CustomPipeline("text-classification", model=model_id, device=-1)


@st.cache_resource(show_spinner="Loading summarizer ...")
def get_summarizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tok, model


def _as_dict(output):
    return output if isinstance(output, dict) else output[0]


def predict_sentiment(text):
    if APP_MODE == "full":
        out = _as_dict(get_sentiment_pipeline(SENTIMENT_MODEL)(
            text, truncation=True, max_length=128))
        out["label"] = SENT_MAP.get(out["label"], out["label"])
    else:
        raw = _as_dict(get_nli_pipeline(SENTIMENT_MODEL)(
            text, candidate_labels=SENTIMENT_LABELS,
            hypothesis_template=SENTIMENT_TEMPLATE))
        out = {"label": raw["labels"][0], "score": raw["scores"][0],
               "metadata": raw.get("metadata")}
    return out


def predict_topic(text):
    out = _as_dict(get_nli_pipeline(ZS_MODEL)(
        text, candidate_labels=TOPICS, hypothesis_template=TOPIC_TEMPLATE))
    return out


def predict_summary(text, tok, model):
    prefix = "summarize: " if "t5" in SUM_MODEL else ""
    inputs = tok(prefix + text, max_length=1024, truncation=True, return_tensors="pt")
    ids = model.generate(**inputs, max_length=80, min_length=20,
                         length_penalty=2.0, num_beams=4)
    return {"summary_text": tok.decode(ids[0], skip_special_tokens=True),
            "metadata": METADATA_VALUE}


def analyze_review(text):
    sent = predict_sentiment(text)
    zs = predict_topic(text)
    summary = None
    if len(text) > 400:
        tok, sum_model = get_summarizer(SUM_MODEL)
        summary = predict_summary(text, tok, sum_model)
    return sent, zs, summary


EXAMPLES = {
    "Custom review": "",
    "Great food, terrible service": "The grilled fish was fresh and delicious, but we waited 45 minutes and the waiter ignored us completely.",
    "Sarcastic short one": "Oh great, another hour waiting for cold food. Just what I wanted.",
    "Neutral mixed": "Kids had fun, however the animals seem under fed, and the cages are too small.",
    "Long Bahraini review": "Tabreez offers a great selection of fish cooked in the traditional Bahraini way. I highly recommend having it with local mashboos rice and the sauce (marag). Definitely don't skip the sauce, as it really takes the flavors to another level. Overall, the food was absolutely delicious. They also have a nice outdoor seating area with a garden vibe, and the service was friendly and efficient. Finally, prices are very reasonable taking into consideration that it is seafood and with huge portions.",
}

st.set_page_config(page_title="Bahrain Review Analyzer", page_icon="🍽️")
st.title("🍽️ Bahrain Visitor-Review Analyzer")
st.caption(f"Pretrained HF models — sentiment · topic (zero-shot) · summary. "
           f"Mode: **{APP_MODE}** (sentiment: `{SENTIMENT_MODEL}`, topics: `{ZS_MODEL}`, "
           f"summary: `{SUM_MODEL}`). Every prediction carries the metadata key "
           "`{'metadata': 'huggingface_AI_model'}`.")

choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Paste a visitor review", value=EXAMPLES[choice], height=160)

if st.button("Analyze", type="primary") and text.strip():
    with st.spinner("Running the models ..."):
        sent, zs, summary = analyze_review(text)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Sentiment")
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent["label"], "❓")
        st.markdown(f"### {emoji} {sent['label']}")
        st.progress(min(float(sent["score"]), 1.0))
        st.caption(f"confidence: {sent['score']:.2%}")
        st.code(str(sent), language=None)

    with c2:
        st.subheader("Topic")
        st.markdown(f"### {zs['labels'][0]}")
        for lab, sc in zip(zs["labels"][:3], zs["scores"][:3]):
            st.write(f"{lab} — {sc:.0%}")
        st.caption("top-3 hypotheses (zero-shot)")
        st.code(f"label: {zs['labels'][0]}\\nscore: {zs['scores'][0]:.4f}\\n"
                f"metadata: '{zs.get('metadata')}'", language=None)

    with c3:
        st.subheader("Summary")
        if summary:
            st.write(summary["summary_text"])
            st.caption(f"metadata: '{summary['metadata']}'")
        else:
            st.write("Review is short enough — no summary needed.")

    with st.expander("Raw pipeline outputs (with the required metadata key)"):
        st.json({"sentiment": sent,
                 "topic": {"sequence": zs["sequence"], "labels": zs["labels"],
                           "scores": zs["scores"], "metadata": zs.get("metadata")},
                 "summary": summary})

st.caption("Lite mode: one shared distilbert-mnli zero-shot model (sentiment + topics) "
           "and t5-small summaries — fits Streamlit Cloud. Set APP_MODE=full to use the "
           "notebook's BERTweet + mDeBERTa + distilbart models. "
           "See sentiment.ipynb for the full evaluation and error analysis.")
