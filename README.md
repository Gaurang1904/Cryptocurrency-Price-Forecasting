# Prediction Model

This repository contains Python scripts and feature engineering workflows for building and evaluating prediction models on cryptocurrency/financial datasets.  
The focus is on preparing data, engineering features, training multiple models, and comparing their performance.

---

## 📂 Project Structure

```
.
├── Data_preparation.py   # Scripts for cleaning and preparing datasets
├── FE.py                 # Feature engineering utilities
├── data.py               # Data loading functions
├── merger.py             # Dataset merging and handling
├── model.py              # Core model training & evaluation
├── model2.py             # Alternative model experiments
├── model3.py             # Extended model variations
├── model4.py             # Additional experiments
├── hello.py              # Test script / basic run
└── .gitignore            # Excluded files and directories
```

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Typical libraries used:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `neuralforecast`

---

## 🚀 Usage

1. **Prepare the data**
   ```bash
   python Data_preparation.py
   ```

2. **Run feature engineering**
   ```bash
   python FE.py
   ```

3. **Train and evaluate a model**
   ```bash
   python model.py
   ```

4. **Experiment with other models**
   ```bash
   python model2.py
   python model3.py
   python model4.py
   ```

---

## 📊 Features

- Data cleaning and preprocessing  
- Feature engineering for time series/financial data  
- Multiple predictive models (baseline + experimental)  
- Evaluation metrics:  
  - Mean Absolute Percentage Error (MAPE)  
  - Directional Accuracy  

---

## 🔮 Future Improvements

- Add support for deep learning models (RNNs, LSTMs, Transformers)  
- Automate hyperparameter tuning  
- Extend evaluation to more asset classes  

---

## 📄 License

This project is licensed under the MIT License.
