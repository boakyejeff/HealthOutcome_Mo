# Missouri Health Outcome Predictor

This repository contains a machine learning pipeline designed to predict health outcomes across various counties in Missouri using socio-demographic datasets. 

> [!WARNING]
> Please note: While this model serves as an educational and exploratory analysis of Missouri's health data, it does not currently exhibit strong statistical correlation or high R-squared values. The results are not intended to be taken as actual predictions for future healthcare outcomes or policy-making.

## Features
- Extracted and cleaned a raw data pipeline from the original Jupyter Notebook into a production-ready Python script (`Final_Project_MO_Health_Story.py`).
- Utilizes `scikit-learn` to build regression forms and evaluate accuracy.
- Visualizes actual vs. predicted metrics utilizing `matplotlib` and `seaborn`.

## Model Performance Visualizations

Below are the scatter plots highlighting the current model's attempt to map predicted vs. actual rates for various health indicators:

| Diabetes | Hypertension |
| --- | --- |
| ![Diabetes](https://github.com/user-attachments/assets/ca1a51fc-9f45-4430-9b0e-b08b8aa72e91) | ![Hypertension](https://github.com/user-attachments/assets/a89dca5f-afed-4fe1-b202-5b37ebf5343e) |

| Obesity | Stroke Hospitalizations |
| --- | --- |
| ![Obesity](https://github.com/user-attachments/assets/98b93890-eacf-42ea-a58b-0c06018591cb) | ![Stroke](https://github.com/user-attachments/assets/b4877712-d124-4358-b778-822a7c2dfa9f) |

## Setup & Execution

It is recommended to use a virtual environment to manage dependencies securely.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/boakyejeff/HealthOutcome_Mo.git
   cd HealthOutcome_Mo
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Script:**
   ```bash
   python Final_Project_MO_Health_Story.py
   ```

## Requirements
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## License
MIT License.
