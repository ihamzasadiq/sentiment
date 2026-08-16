# Pretrained Model Challenge

## Hands-on Lab: Discovering, Evaluating, and Using Hugging Face Models

**Student Lab Instructions**

- **Author:** Ali El-Kassas
- **Level:** Intermediate
- **Theme:** Using, inspecting, evaluating, and applying pretrained models from Hugging Face

In this lab, you will work with pretrained models available on the Hugging Face Hub. Instead of training a model from scratch, you will learn how to select an appropriate model, run inference, inspect what happens inside the model, evaluate its predictions, compare models, and build a small AI application.

The central question of this lab is: *"Someone has already trained a large model. How can we intelligently use it, evaluate it, and decide whether we need to adapt it?"*

---

## Learning Objectives

- Understand what a pretrained model is and why pretrained models are useful.
- Navigate the Hugging Face Model Hub and select an appropriate model.
- Read and interpret a model card, including task, architecture, training data, language, license, and limitations.
- Use the Transformers library to perform inference with a pretrained model.
- Understand the basic flow from input → tokenizer/processor → model → logits/prediction.
- Evaluate a pretrained model on a small real-world dataset.
- Measure model performance using appropriate evaluation metrics.
- Perform error analysis and identify situations where a pretrained model fails.
- Compare multiple pretrained models and make an evidence-based model-selection decision.
- Understand when zero-shot inference is useful.
- Understand why domain shift can cause pretrained models to perform poorly.
- Recognize when fine-tuning or another adaptation strategy may be necessary.
- Combine pretrained models into a simple AI application.

---

## The Core Idea

The central question of this lab is:

> "I already have a powerful pretrained model. How can I intelligently use it, evaluate it, and decide whether I need to adapt it?"

---

## Lab Scenario

Imagine that a client wants to understand visitor reviews (restaurants, hotels, etc.) in Bahrain. Your task is to build a small AI system that can analyze review text. Don't start with training a model from scratch.

Instead, you must investigate what is already available on Hugging Face and determine whether an existing model can solve your problem.

The final system should be able to answer questions such as:

- Is the review positive, negative, or otherwise classified by the selected model?
- What topic is the review mainly about: food, service, price, location, or ambience?
- Can the review be summarized automatically?
- Where does the pretrained model make mistakes?
- Should the company use the pretrained model as-is, or should it fine-tune a model on local data?

---

## Rules

- Do not start by training a neural network from scratch.
- You must use at least three pretrained models from the Hugging Face Model Hub.
- You must collect or create your own evaluation data.
- You must evaluate the model instead of showing only a few example predictions.
- You must document model failures.
- You must justify your model choice using evidence.
- If the first model performs poorly, investigate why before immediately replacing it.

---

## Task A — Text Investigation

Choose a text domain that interests you. Examples include restaurant reviews, hotel reviews, movie reviews, product reviews, customer complaints, or app reviews.

### Suggested Workflow

1. Collect 50–100 text samples from appropriate public sources or create a small labeled dataset yourself.
2. Define the task clearly: sentiment, emotion, or another classification problem.
3. Find one or more suitable pretrained models on Hugging Face. Your pipeline must output a custom dictionary structure. In addition to the standard Hugging Face label and score, your code must append a custom metadata key named `{'metadata': 'huggingface_AI_model'}`.
4. Run inference on every review.
5. Calculate performance metrics.
6. Investigate incorrect predictions.
7. Compare at least three models if possible.
8. Explain which model you would choose and why.

---

## Task B — Zero-Shot Classification

Define the candidate classes yourself and use a pretrained zero-shot model.

**Example:** classify e-commerce reviews into:

- Wrong Product
- Damaged Product
- Late Delivery
- Customer Support
- Refund Request
- Payment Problem
- Product Quality
- Missing Item

**Another example:** classify restaurant reviews into:

- Food Quality
- Service
- Price
- Cleanliness
- Atmosphere
- Location
- Waiting Time

Create 30–50 manually labeled examples. Run the zero-shot model and compare its predictions with your labels. Your pipeline must output a custom dictionary structure. In addition to the standard Hugging Face label and score, your code must append a custom metadata key named `{'metadata': 'huggingface_AI_model'}`. Calculate accuracy and/or F1 score, then investigate the failures.

---

## Step 1 — Define Your Problem

Before searching for a model, write a precise problem statement. Answer:

- What problem are you solving?
- Who would use the solution?
- What is the input?
- What should the model produce?
- How will you know whether the model is successful?

---

## Step 2 — Collect Your Data

You are responsible for creating the evaluation dataset. A small dataset is enough for this lab; quality and clear labeling are more important than having thousands of examples.

**Recommended minimum:**

- Text/image classification: 50–100 examples
- Audio: 20–50 recordings
- Zero-shot classification: 30–50 manually labeled examples

Keep a record of where the data came from. Do not collect private, sensitive, or copyrighted material in ways that violate the source's terms.

---

## Step 3 — Find a Pretrained Model on Hugging Face

Search the Hugging Face Model Hub for models that match your task. Do not choose a model only because it has a large number of downloads.

### Model Card Investigation

- Model name
- Author or organization
- Task
- Architecture
- Training dataset
- Languages
- Number of parameters
- License
- Intended use
- Known limitations
- Evaluation results

---

## Step 4 — Run Your First Inference

Experiment with simple, ambiguous, long, short, positive, negative, and contradictory examples. Record interesting predictions.

---

## Step 6 — Evaluate the Model

Do not evaluate the model using only a handful of examples. Create a labeled evaluation set and measure performance.

**Possible metrics:**

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Inference time
- Human evaluation for generated text

Choose metrics appropriate to your task and explain why.

---

## Step 7 — Error Analysis

This is one of the most important parts of the lab. Do not simply report the score. Study where the model fails.

For each interesting failure, record:

- Input
- Expected output
- Model prediction
- Confidence/score if available
- Your explanation of the failure

Look for patterns such as language differences, ambiguity, sarcasm, spelling mistakes, poor image quality, background noise, domain-specific terminology, or classes that are visually/textually similar.

---

## Step 8 — Model Battle

Choose at least two pretrained models for the same task. Run both models on the same evaluation data.

**Compare:**

- Accuracy/F1 or another appropriate metric
- Inference speed
- Model size
- Language support
- License
- Known limitations
- Failure cases

The winner is not necessarily the model with the highest accuracy. Make a recommendation based on the actual application requirements.

---

## Step 9 — Ask: Does the Model Understand My Domain?

Compare the model's original training domain with your data. Identify possible domain shift.

For example, a generic image model may recognize common objects but struggle with local food categories. A general English sentiment model may struggle with Arabic, dialect, sarcasm, or domain-specific language.

---

## Step 10 — Decide What to Do Next

Based on your results, decide whether the pretrained model is good enough or whether it needs improvement.

- Use the pretrained model as-is.
- Try a different pretrained model.
- Improve preprocessing.
- Change the prompt or candidate labels for zero-shot tasks.
- Use a larger or better evaluation dataset.
- Fine-tune the model on your domain-specific data.

---

## The Model Lifecycle You Should Discover

Your investigation should lead to the following general workflow:

```
Problem → Collect Data → Find Pretrained Model → Read Model Card → Inference →
Evaluation → Error Analysis → Compare Models → Decide: Use / Replace / Adapt
```

---

## Final Project — Build a Small Application

Turn your experiment into a simple application. A Streamlit interface is recommended, but another lightweight interface is acceptable.

**Possible applications:**

- Review sentiment analyzer
- News topic classifier
- Local food image classifier
- Audio transcription tool
- Tourist review analyzer
- Zero-shot document classifier
- Multilingual text analyzer

---

## Final Deliverables

- Jupyter notebook containing the complete experiment.
- Your collected dataset or a documented description of how it was collected.
- Model-selection investigation.
- Evaluation results.
- Error analysis with at least 10 interesting failure cases where applicable.
- Comparison of at least two models where feasible.
- A working small application or demo.
- A short report explaining your conclusions.

---

## Short README.md Template

1. Problem Definition
2. Data Collection
3. Data Preparation
4. Model Selection
5. Model Card Investigation
6. Initial Inference Results
7. Evaluation
8. Error Analysis
9. Model Comparison
10. Limitations
11. Final Recommendation
12. Future Work / Fine-Tuning Proposal

---

## Challenge Extension — Fine-Tuning

Only after completing the pretrained-model investigation should you consider fine-tuning. If the model performs poorly on your domain, propose how you would adapt it using your collected dataset.

Answer:

- Why is the pretrained model failing?
- What additional training data would you need?
- Which layers or components would you fine-tune?
- How would you compare the fine-tuned model with the original model?

---

## Assessment Rubric

| Component | Weight | What is Assessed? |
|---|---|---|
| Problem & Data | 15% | Clear problem, appropriate data, quality and documentation |
| Model Selection | 15% | Appropriate Hugging Face model and strong model-card investigation |
| Implementation | 20% | Correct use of pretrained model and reproducible code |
| Evaluation | 15% | Appropriate metrics and meaningful results |
| Error Analysis | 15% | Depth of investigation into model failures |
| Model Comparison | 10% | Evidence-based comparison and recommendation |
| Application & Presentation | 10% | Working demo and clear communication |

---

## Reflection Questions

- What surprised you about the pretrained model? *(Answer all the questions with "I think")*
- What types of examples caused the most failures?
- Did a larger or more popular model necessarily perform better?
- How important was the training data of the pretrained model?
- What did you learn from the model card?
- What is the difference between using a pretrained model and training a model from scratch?
- When would fine-tuning be worth the additional effort?
- If you had 10 times more data, what would you change?
- Would you deploy your model in a real application? Why or why not?
