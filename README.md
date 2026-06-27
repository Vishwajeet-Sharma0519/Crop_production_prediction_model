# Crop Production Prediction: End-to-End Machine Learning Pipeline

An End-to-End Machine Learning project focused on predicting agricultural crop production across various Indian districts using historical crop yield data combined with geographical rainfall and climatic weather data. 

This project demonstrates a complete Data Science lifecycle: from raw data gathering and rigorous cleansing, to advanced preprocessing, hyperparameter tuning across multiple models (Linear Regression, ANN, Random Forest, XGBoost), and finally deploying the winning model into an interactive web application.

# Datasets Used
1. **Crop Production**: Historical data of crop yields across Indian States and Districts (1997 - 2015).
2. **Rainfall Normal**: District-wise historical normal average rainfall across different months.
3. **Weather Data**: Historical average temperatures (`tavg`, `tmin`, `tmax`) and daily precipitation (`prcp`).

# Tech Stack
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Pipelines, Target Encoding, Random Forest), XGBoost
- **Deep Learning**: TensorFlow / Keras (Artificial Neural Networks)
- **Web Deployment**: Streamlit
- **Environment**: Jupyter Notebook

# The End-to-End Workflow

### 1. Data Collection & Cleansing
Agriculture data presents unique challenges with massive scale variances (e.g., spices vs. sugarcane). 
- **Merging**: Unified historical crop data with localized climate data to provide the model with environmental context.
- **Cleansing**: Removed extreme outliers, dropped crops with statistically insignificant data points (<9 records), and filtered out microscopic production values (<10 tonnes) to prevent the models from heavily biasing towards zeroes.

### 2. Advanced Preprocessing Pipeline
To prevent data leakage and ensure seamless deployment, all transformations were bundled into a master `ColumnTransformer` utilizing internal Scikit-Learn `Pipeline`s:
- **Numerical Imputation & Scaling**: Missing values in continuous features were intelligently filled using `mean` (for total rainfall) or `median` (for temperatures) via `SimpleImputer`, followed strictly by `StandardScaler`.
- **High-Cardinality Encoding**: Deployed **Target Encoding** for heavily grouped categorical variables (`District_Name`, `State_Name`, `Crop`) to capture complex historical patterns without exploding the dataset's dimensionality.
- **Low-Cardinality Encoding**: Applied standard **One-Hot Encoding** for features like `Season`.

### 3. Model Training & Hyperparameter Tuning
We trained and evaluated a diverse suite of models to capture both linear trends and complex, non-linear environmental interactions:
- **Regularized Linear Regression (Ridge)**: Tuned via `GridSearchCV` to establish a baseline.
- **Artificial Neural Network (ANN)**: Built a Multi-Layer Perceptron architecture utilizing `BatchNormalization`, `Dropout`, and learning rate scheduling (`ReduceLROnPlateau`).
- **Random Forest**: Tuned using `RandomizedSearchCV` (`n_jobs=-1`) to prevent deep-tree overfitting while maximizing CPU efficiency.
- **XGBoost**: Utilized histogram-based tree building (`tree_method='hist'`) for lightning-fast, highly optimized gradient boosting.

### 4. Model Comparison & Selection
The non-linear interactions in crop production proved too complex for linear models. **XGBoost** emerged as the superior model, balancing incredibly fast training times with the highest R² score and lowest Root Mean Squared Error (RMSE). 

### 5. Deployment
The winning XGBoost model was appended directly to the end of our preprocessing `ColumnTransformer` to create a single, unified inference pipeline. This pipeline was serialized using `joblib` and deployed via a dynamic **Streamlit** web application. 

The web app intelligently filters dropdowns (e.g., mapping specific districts to their respective states dynamically) and handles all raw user input—passing it directly to the pipeline for real-time predictions.

---

## How to Run the Web App

1. Clone the repository and ensure you have the necessary datasets in the root directory.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn streamlit
   ```
3. Open a terminal in the project directory and launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Access the clean, interactive web interface via your local browser to predict crop production!

## Exported Assets
- `app.py`: The main Streamlit web application.
- `production_xgboost_pipeline.pkl`: The fully bundled Scikit-Learn pipeline containing both the data preprocessor and the fitted XGBoost regressor, ready for raw inference.
