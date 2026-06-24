# QA 3 (Unit 3): Naive Bayes — Manual + Code Implementation

| | |
|---|---|
| **Name** | Monica R |
| **Class** | 2nd MSIC |
| **Reg No** | 25MSIC103 |
| **Topic** | Naive Bayes Classifier |
| **Dataset** | Spam / Ham Text Messages |

---

## 📁 Repository Structure

```
qa3-naive-bayes/
│
├── naive_bayes_qa3.ipynb        # Jupyter Notebook (Manual + sklearn)
├── naive_bayes_explanation.docx # Word document explaining all steps
└── README.md                    # This file
```

---

## 📌 What is Naive Bayes?

Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**:

```
P(C | X) = P(X | C) × P(C) / P(X)
```

It assumes all features (words) are **conditionally independent** given the class — the "naive" assumption. Despite this simplification, it performs very well for text classification tasks like spam detection.

### Key Concepts

| Term | Formula | Meaning |
|---|---|---|
| **Prior** P(C) | count(C) / N | How often each class appears in training data |
| **Likelihood** P(w\|C) | (count(w,C)+1) / (total_C+V) | Word frequency per class (Laplace smoothed) |
| **Posterior** P(C\|X) | P(C) × ∏ P(wi\|C) | Final score used to make prediction |

---

## 📓 Notebook Overview (`naive_bayes_qa3.ipynb`)

### Part 1 — Manual Naive Bayes

Step-by-step manual calculation for classifying `"free money today"`:

1. **Step 1** — Count documents per class → compute Priors
2. **Step 2** — Build vocabulary & word frequency table
3. **Step 3** — Compute likelihoods with Laplace smoothing (+1)
4. **Step 4** — Multiply prior × likelihoods for each class
5. **Step 5** — Normalize scores → predict class

**Result:** `"free money today"` → **SPAM (≈64%)**

### Part 2 — sklearn Implementation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train    = vectorizer.fit_transform(messages)

clf = MultinomialNB(alpha=1.0)   # alpha=1.0 → Laplace smoothing
clf.fit(X_train, labels)

X_test = vectorizer.transform(["free money today"])
print(clf.predict(X_test))        # → ['spam']
print(clf.predict_proba(X_test))  # → probabilities per class
```

### Comparison Table

| Method | P(SPAM) | P(HAM) | Prediction |
|---|---|---|---|
| Manual (Step-by-Step) | ≈ 64.0% | ≈ 36.0% | **SPAM** |
| sklearn MultinomialNB | ≈ 64.1% | ≈ 35.9% | **SPAM** ✅ |

Both methods agree — confirming the manual calculation is correct.

---

## 🗂️ Dataset

### Training Data (6 messages)

| Message | Label |
|---|---|
| free money win prize | SPAM |
| win free lottery now | SPAM |
| free prize offer win | SPAM |
| meeting at noon today | HAM |
| lunch today at office | HAM |
| please call me today | HAM |

### Test Messages (5 messages)

| Message | Prediction |
|---|---|
| free money today | SPAM |
| call me at office | HAM |
| win free prize now | SPAM |
| lunch meeting today | HAM |
| lottery offer free | SPAM |

---

## ⚙️ How to Run

### Requirements

```bash
pip install scikit-learn numpy pandas jupyter
```

### Run the Notebook

```bash
jupyter notebook naive_bayes_qa3.ipynb
```

Or open directly in **VS Code**, **Google Colab**, or **JupyterLab**.

---

## 📄 Word Document

`naive_bayes_explanation.docx` contains:

- Explanation of Naive Bayes and Bayes' Theorem
- Definition of Prior and Likelihood with formulas
- Full manual solution (5 steps with tables)
- sklearn code walkthrough
- Manual vs sklearn comparison
- Multi-message results
- Summary table

---

## 🧠 Why Naive Bayes Works for Text

- Fast to train — **O(n)** time complexity
- Works well with **small datasets**
- Handles **high-dimensional** feature spaces (large vocabularies)
- Provides **probability scores**, not just labels
- Robust to irrelevant features

---

*QA 3 — Unit 3 | 2nd MSIC | Monica R | Reg No: 25MSIC103*
