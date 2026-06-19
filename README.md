# LSTM Based Sentiment Analysis

This repository contains a Korean text emotion-classification experiment using
an LSTM model. It includes the training notebook, a saved Keras model and a
small inference script.

## Files

| Path | Description |
| --- | --- |
| `LSTM.ipynb` | Main training notebook |
| `5차년도_2차.csv` | Training data used by the notebook |
| `model.h5` | Saved Keras model |
| `model_test.py` | Simple interactive inference script |

## Workflow

The notebook follows this pipeline:

1. Load text data with Pandas.
2. Tokenize sentences with TensorFlow/Keras `Tokenizer`.
3. Pad sequences to a fixed length.
4. Train an Embedding + LSTM classifier.
5. Save the model as `model.h5`.

Notebook output shows validation accuracy reaching roughly the mid-80% range
during training. This should be treated as an experiment log rather than a
fully reproduced benchmark.

## How To Run

```bash
pip install -r requirements.txt
jupyter notebook LSTM.ipynb
```

For inference:

```bash
python model_test.py
```

## Important Note

`model_test.py` loads `model.h5`, but the fitted tokenizer is not saved in this
repository. For reliable inference, save the tokenizer during training and load
the same tokenizer in `model_test.py`.

## Suggested Next Step

Add a `tokenizer.pkl` or `tokenizer.json` export from the notebook, then update
`model_test.py` to load it before prediction.
