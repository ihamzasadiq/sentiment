# Bahrain Visitor Review Analyzer

## 1. Problem Definition

This lab evaluates pretrained Hugging Face models for visitor-review analysis in Bahrain. The goal is to decide whether existing models can classify review sentiment, identify the main review topic with zero-shot classification, and summarize longer reviews without training a neural network from scratch.

The intended user is an analyst or small tourism/restaurant business owner who wants a fast first pass over public visitor feedback. The input is free-text review content. The expected outputs are:

- sentiment: `positive`, `negative`, or `neutral`
- topic: `Food Quality`, `Service`, `Price`, `Location`, or `Ambience`
- optional summary for long reviews
- the required metadata field: `metadata: huggingface_AI_model`

Success is measured with accuracy, macro-F1, confusion matrices, inference time, and qualitative error analysis.

## 2. Data Collection

The sentiment evaluation set is stored in `reviews.csv` and contains 200 Bahrain visitor reviews with three manually assigned sentiment labels. The topic set is stored in `topics_labeled.csv` and contains 40 manually labeled examples, balanced across five topic classes.

Sentiment distribution:

| Label | Count |
|---|---:|
| positive | 143 |
| negative | 49 |
| neutral | 8 |

Topic distribution:

| Topic | Count |
|---|---:|
| Food Quality | 8 |
| Service | 8 |
| Price | 8 |
| Location | 8 |
| Ambience | 8 |

The sentiment data is imbalanced toward positive reviews, and the neutral class is small. The notebook reports one duplicated review. These limitations matter because high accuracy can hide weak neutral-class performance.

## 3. Data Preparation

The notebook reads the CSV files directly with pandas. No model training or fine-tuning is performed. Cached prediction files in `results/` are used by default so the notebook does not rerun expensive inference unless `force=True` is passed to the helper functions.

Review text is passed to Hugging Face pipelines with truncation enabled for the sentiment models. Zero-shot topic classification uses candidate labels and hypothesis templates rather than task-specific training.

## 4. Model Selection

Three pretrained sentiment models were evaluated:

| Model | Purpose |
|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Twitter/X-style sentiment classification |
| `finiteautomata/bertweet-base-sentiment-analysis` | Tweet sentiment classification |
| `nlptown/bert-base-multilingual-uncased-sentiment` | Multilingual star-rating sentiment converted to 3 labels |

Zero-shot topic classification used:

| Model | Purpose |
|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Higher-quality multilingual NLI zero-shot classifier |
| `typeform/distilbert-base-uncased-mnli` | Faster lightweight NLI zero-shot baseline |

Summarization used `sshleifer/distilbart-cnn-12-6` in the notebook and `t5-small` in Streamlit lite mode.

## 5. Model Card Investigation

The notebook includes a Hugging Face Hub metadata lookup for model author, downloads, likes, license, and library when network access is available. The model comparison also records practical factors such as model size, inference time, training style, and license.

Important model-card observations:

- Twitter-trained models are a plausible fit for short public reviews but may struggle with local phrasing, mixed sentiment, sarcasm, or longer review structure.
- The multilingual star-rating model supports broader language coverage, but mapping star labels to `positive`, `negative`, and `neutral` loses nuance.
- Zero-shot NLI models do not learn the domain labels from this dataset; performance depends heavily on label names and hypothesis wording.
- The topic set is small, so zero-shot prompt improvements may overfit the manually assigned labels.

## 6. Initial Inference Results

The notebook tests simple, ambiguous, sarcastic, mixed, long, and local-language-flavored examples. The custom wrapper adds the required field to each Hugging Face pipeline output:

```python
{"metadata": "huggingface_AI_model"}
```

The notebook also includes an assertion confirming the field appears in sentiment output, and the Streamlit app displays the metadata in the visible result panels and raw JSON output.

## 7. Evaluation

Cached predictions in `results/` produce the following sentiment metrics on 200 reviews:

| Model | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Inference Time | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `twitter-roberta-base-sentiment-latest` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 32.3s | 124.6M |
| `bertweet-base-sentiment-analysis` | 0.910 | 0.715 | 0.964 | 0.880 | 0.300 | 36.3s | 134.9M |
| `bert-base-multilingual-uncased-sentiment` | 0.825 | 0.611 | 0.917 | 0.791 | 0.125 | 32.9s | 167.4M |

Cached zero-shot topic metrics on 40 topic-labeled reviews:

| Model / Template | Accuracy | Macro-F1 | Top-2 Accuracy | Inference Time |
|---|---:|---:|---:|---:|
| mDeBERTa, `This review is about {}.` | 0.400 | 0.368 | 0.800 | ~204s |
| mDeBERTa, `The main topic of this review is {}.` | 0.575 | 0.576 | 0.825 | ~211s |
| distilbert-mnli, `This review is about {}.` | 0.325 | 0.302 | 0.650 | ~9s |

## 8. Error Analysis

The strongest sentiment result is `twitter-roberta-base-sentiment-latest`, but the perfect score should be treated cautiously because the dataset is small and manually labeled. The other two sentiment models expose the main failure modes:

- neutral reviews are rare, so neutral F1 is weak for BERTweet and NLP Town
- mixed reviews with praise and complaints are difficult to assign to one sentiment
- sarcastic reviews can be misread because the literal words may sound positive
- long reviews may contain multiple experiences and conflicting sentiment
- local names, dialect terms, and Bahrain-specific context create domain shift

For topic classification, the main topic is often ambiguous. A review can mention food, price, service, and ambience at the same time. The improved mDeBERTa template performs better than the default wording, but 17 of 40 topic examples are still wrong. Many errors are near misses where the correct label appears in the top two candidates.

## 9. Model Comparison

For sentiment, `cardiffnlp/twitter-roberta-base-sentiment-latest` is the best current choice because it has the highest cached accuracy and macro-F1 with competitive inference time. BERTweet is a reasonable backup but performs poorly on the neutral class. NLP Town is useful for multilingual/star-rating comparison but loses detail when star labels are collapsed into three sentiment classes.

For topics, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` with the improved template is the best quality choice. `typeform/distilbert-base-uncased-mnli` is much faster and is therefore useful for the lightweight Streamlit demo, but it is less accurate on the topic set.

## 10. Limitations

- The sentiment set is imbalanced, with only 8 neutral examples.
- The topic set has only 40 examples, so metrics have high variance.
- Some topic labels overlap naturally, especially food quality vs service and price vs food quality.
- Cached model results should be refreshed only when model versions, dependencies, or data change.
- The lab does not verify whether all review text can be redistributed under the original source terms.
- The Streamlit demo is a lightweight educational interface, not a monitored production service.

## 11. Final Recommendation

Use `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment classification. Use `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` with the template `The main topic of this review is {}.` for higher-quality topic classification. For lightweight demos or constrained hosting, use the Streamlit lite mode with `typeform/distilbert-base-uncased-mnli`, while clearly labeling it as the faster but less accurate option.

The company should not fine-tune immediately. It should first collect more balanced local labels, especially neutral reviews and topic examples with clear annotation rules.

## 12. Future Work / Fine-Tuning Proposal

Fine-tuning would be worth considering after collecting a larger Bahrain-specific dataset with balanced labels and clear annotation guidelines. The next dataset should include sarcasm, Arabizi/local terms, mixed sentiment, short comments, and long multi-topic reviews.

For sentiment, fine-tune a compact transformer classifier and compare it against `twitter-roberta-base-sentiment-latest`. For topics, either fine-tune a supervised topic classifier or keep zero-shot classification but improve label descriptions and allow multi-label outputs where appropriate.

## Reproducibility

Use Python 3.11 or 3.12. Python 3.14 may not have matching PyTorch wheels on the default package index.

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `sentiment.ipynb` in Jupyter, VS Code, Colab, or another notebook environment after installing the requirements.

The app starts in lite mode by default:

```bash
streamlit run app.py
```

To use the heavier notebook-style models:

```bash
APP_MODE=full streamlit run app.py
```

The notebook and app use cached prediction files in `results/` by default. Setting `force=True` in the notebook helper functions recomputes predictions and can trigger expensive model downloads and inference.
