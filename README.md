# Data Mining Project: Classification Competition

## Project Overview
Welcome to our data mining project for the Kaggle Playground Series, Season 4, Episode 11! This project involves building a robust classification model to predict target variables based on structured data provided in the competition dataset. We aim to achieve high accuracy while implementing efficient data preprocessing, feature engineering, and model optimization techniques. 

## Competition Information
- **Kaggle Competition**: [Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11/overview)
- **Goal**: Classify samples into categories using a labeled dataset.
- **Evaluation Metric**: Logarithmic Loss (LogLoss)

## Project Structure

The repository structure is organized as follows:
```
data-mining-project/
├── data/
│   ├── raw/                # Raw competition data
│   ├── processed/          # Processed data ready for modeling
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Notebook for data cleaning and preprocessing
│   ├── 02_eda.ipynb                   # Notebook for exploratory data analysis (EDA)
│   ├── 03_feature_engineering.ipynb   # Notebook for feature engineering and selection
│   ├── 04_model_training.ipynb        # Notebook for model training and optimization
│   ├── 05_evaluation.ipynb            # Notebook for model evaluation and interpretation
├── src/
│   ├── data_preprocessing.py          # Python script for data preprocessing
│   ├── feature_engineering.py         # Script for feature engineering
│   ├── train_model.py                 # Script for model training
│   └── evaluate_model.py              # Script for model evaluation
└── README.md
```

## Project Phases

1. **Data Collection**: Download and store the dataset provided by the competition organizers in the `data/raw/` directory.
  
2. **Data Preprocessing**: Handle missing values, outliers, and categorical encoding in the `data_preprocessing.py` script and the associated Jupyter notebook.

3. **Exploratory Data Analysis (EDA)**: Visualize the distribution of features, investigate correlations, and analyze patterns to inform feature engineering decisions.

4. **Feature Engineering**: Create and select features that may enhance the performance of the model. We use dimensionality reduction and select important features based on their correlations with the target variable.

5. **Model Training**: Develop a classification model using machine learning algorithms XgGBoost.

6. **Model Evaluation**: Assess model performance using cross-validation and the competition's evaluation metric (LogLoss). Record results for various models and iterations in the `evaluation.ipynb` notebook.

7. **Submission**: After fine-tuning, generate predictions on the test set and submit the results on Kaggle.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/KevinUnedo/project-DaMi-05.git
   cd project-DaMi-05
   ```

## Usage

1. **Data Preprocessing**:
   - Run the `data_preprocessing.py` script or execute the cells in `01_data_preprocessing.ipynb`.

2. **Exploratory Data Analysis**:
   - Open and run the `02_eda.ipynb` notebook.

3. **Feature Engineering**:
   - Execute the `03_feature_engineering.ipynb` notebook or run the `feature_engineering.py` script.

4. **Model Training**:
   - Train the model by running the `04_model_training.ipynb` notebook or the `train_model.py` script.

5. **Model Evaluation**:
   - Evaluate model performance using the `05_evaluation.ipynb` notebook or `evaluate_model.py` script.

6. **Submission**:
   - Generate predictions on the test set for Kaggle submission.

## Results and Performance

Results, accuracy scores, and model comparisons will be recorded here. The final model will be selected based on the lowest LogLoss score achieved on the validation set.

## Acknowledgments

Special thanks to Kaggle for hosting the competition and providing the dataset.

## Contributors

- 12S21019 [Alex Mario Kristian](https://github.com/AlexMari0)
- 12S21002 [Sitogab Antonio Girsang](https://github.com/SitogabAntonio)
- 12S21016 [Kevin Unedo Samosir](https://github.com/KevinUnedo)
