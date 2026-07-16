# Sector Classifier

An offline NLP project that predicts an organization's **Sector** and
**Subsector** from its textual **Definition**, using TF-IDF features and
LinearSVC classifiers. No external APIs are used — everything runs
locally.

## Project Overview

Given a free-text description of an organization, the system predicts:

1. **Sector** (e.g. `Energy`, `Finance`, `Healthcare`)
2. **Subsector** (e.g. `Power`, `Banking`, `Healthcare Services`)

Two independent `LinearSVC` classifiers are trained on TF-IDF features
extracted from cleaned, lemmatized definition text.

## Folder Structure

```
SectorClassifier/
│
├── data/
│   └── Sector_Classifier_Definition_Data.xlsx
│
├── models/
│   ├── tfidf.pkl
│   ├── sector_model.pkl
│   └── subsector_model.pkl
│
├── train.py
├── predict.py
├── app.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

NLTK resources (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) are
downloaded automatically the first time `utils.py` is imported.

## Training

Place the training Excel file at `data/Sector_Classifier_Definition_Data.xlsx`
(with columns `Definition`, `Sector_of_Org`, `Sub_Sector`), then run:

```bash
python train.py
```

This will:
- Load and clean the dataset
- Preprocess text (lowercase, punctuation removal, tokenization,
  stopword removal, lemmatization)
- Vectorize text with TF-IDF
- Train a `LinearSVC` model for Sector and another for Subsector
  (80/20 stratified split, `random_state=42`)
- Print Accuracy, Precision, Recall, F1 Score, Classification Report,
  and Confusion Matrix for each model
- Save `tfidf.pkl`, `sector_model.pkl`, and `subsector_model.pkl` to
  `models/`

Note: classes with fewer than 2 samples cannot be stratified and are
automatically excluded from that specific split, with a warning printed
to the console.

## Prediction

```python
from predict import predict

result = predict(
    "A government body responsible for regulating electricity "
    "generation, transmission, and distribution across the country."
)
print(result)
# {'sector': 'Energy', 'subsector': 'Power'}
```

Or run the script directly for a sample prediction:

```bash
python predict.py
```

## Running the Streamlit App

```bash
streamlit run app.py
```

The app has two tabs:

- **Single Prediction** — enter an organization's definition in the text
  area and click **Predict** to see the predicted Sector and Subsector.
- **Batch from Organization Names** — upload an Excel file containing a
  list of organization names, and the app fetches each organization's
  definition from Wikipedia, predicts Sector/Subsector, shows the
  results in a table, and lets you download them as an Excel file.
  Requires an internet connection.

## Batch Prediction from Organization Names (CLI)

If you have a list of organization names (e.g. 50 organizations) and
want an updated Excel with their web-sourced definitions and predicted
Sector/Subsector, use `batch_predict.py` instead of the Streamlit app:

```bash
python batch_predict.py --input organizations.xlsx --output results.xlsx
```

- `--input`: Excel file with organization names in one column (any
  column name; the first column is used by default).
- `--column`: Optional. Specify the exact column name to read names from,
  e.g. `--column "Organization Name"`.
- `--output`: Optional. Path for the output Excel file (default:
  `batch_results.xlsx`).
- `--delay`: Optional. Seconds to wait between Wikipedia requests
  (default: `0.5`), to avoid hitting rate limits.

The output Excel contains:

| Organization_Name | Definition | Source | Sector | Subsector |
|---|---|---|---|---|

`Source` indicates where the definition came from: `wikipedia_direct`
(exact page match), `wikipedia_search` (best search match), or
`not_found` (no Wikipedia page located — Sector/Subsector left blank).

**Notes:**
- This step requires internet access to reach Wikipedia; everything
  else in the project (training, prediction) remains fully offline.
- Smaller or private organizations without a Wikipedia page will not
  get a definition and are left blank — you'd need to supply their
  definitions manually in those cases.
- Definitions are pulled from the opening summary of the matching
  Wikipedia article.

## Example Output

```
Definition: A government body responsible for regulating electricity
generation, transmission, and distribution across the country.
Predicted Sector   : Energy
Predicted Subsector: Power
```

## Technologies Used

- Python 3
- pandas, numpy
- scikit-learn (TF-IDF, LinearSVC, train/test split, metrics)
- NLTK (tokenization, stopwords, lemmatization)
- joblib (model persistence)
- Streamlit (UI)
- openpyxl (Excel I/O)
