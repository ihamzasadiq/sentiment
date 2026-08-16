import logging

import streamlit as st
from transformers import Pipeline, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.getLogger("transformers.utils.loading_report").setLevel(logging.CRITICAL)

SENTIMENT_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"
ZS_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"
TOPICS = ["Food Quality", "Service", "Price", "Location", "Ambience"]
ZS_TEMPLATE = "The main topic of this review is {}."
SENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
METADATA_VALUE = "huggingface_AI_model"


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


@st.cache_resource(show_spinner="Loading sentiment model ...")
def get_sentiment_pipeline():
    return CustomPipeline("text-classification", model=SENTIMENT_MODEL, device=-1)


@st.cache_resource(show_spinner="Loading topic model ...")
def get_zero_shot_pipeline():
    return CustomPipeline("zero-shot-classification", model=ZS_MODEL, device=-1)


@st.cache_resource(show_spinner="Loading summarizer ...")
def get_summarizer():
    tok = AutoTokenizer.from_pretrained(SUM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)
    return tok, model


def analyze_review(text, sent_pipe, zs_pipe, tok, sum_model):
    sent_out = sent_pipe(text, truncation=True, max_length=128)
    sent = sent_out if isinstance(sent_out, dict) else sent_out[0]
    sent["label"] = SENT_MAP.get(sent["label"], sent["label"])

    zs_out = zs_pipe(text, candidate_labels=TOPICS, hypothesis_template=ZS_TEMPLATE)
    zs = zs_out if isinstance(zs_out, dict) else zs_out[0]

    summary = None
    if len(text) > 400:
        inputs = tok(text, max_length=1024, truncation=True, return_tensors="pt")
        ids = sum_model.generate(**inputs, max_length=80, min_length=20,
                                 length_penalty=2.0, num_beams=4)
        summary = {
            "summary_text": tok.decode(ids[0], skip_special_tokens=True),
            "metadata": METADATA_VALUE,
        }
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
st.caption("Pretrained HF models — sentiment (BERTweet) · topic (zero-shot mDeBERTa) · "
           "summary (distilbart). Every prediction carries the metadata key "
           "`{'metadata': 'huggingface_AI_model'}`.")

choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Paste a visitor review", value=EXAMPLES[choice], height=160)

if st.button("Analyze", type="primary") and text.strip():
    sent_pipe = get_sentiment_pipeline()
    zs_pipe = get_zero_shot_pipeline()
    tok, sum_model = get_summarizer()

    with st.spinner("Running the models ..."):
        sent, zs, summary = analyze_review(text, sent_pipe, zs_pipe, tok, sum_model)

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
        st.json({"sentiment": sent, "topic": {"sequence": zs["sequence"],
                                               "labels": zs["labels"],
                                               "scores": zs["scores"],
                                               "metadata": zs.get("metadata")},
                 "summary": summary})

st.caption("Models: BERTweet (sentiment, 0.71 macro-F1 on our 200-review test set) · "
           "mDeBERTa-v3 zero-shot NLI (topics) · distilbart-cnn-12-6 (summaries). "
           "See sentiment.ipynb for the full evaluation and error analysis.")
