# Email Spam Classification Project

## Problem Statement
The goal of this project is to develop a predictive model for classifying incoming email messages as either spam or not spam. The dataset used contains information on 5,172 email files, each labeled with 0 for not spam and 1 for spam. The dataset includes 3,000 columns representing the 3,000 most common words in the emails.

## Dataset
The dataset used in this project can be found [here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data). It consists of email information, including the labels for spam or not spam, and features representing the most common words in the emails.

## Libraries and Tools
- `numpy`, `pandas`: Data manipulation and analysis
- `seaborn`, `matplotlib.pyplot`: Data visualization
- `scikit-learn`: Machine learning models and tools
- `pgmpy`: Library for Bayesian Network modeling

## Data Preprocessing
- Removed unnecessary columns and handled missing values.
- Explored the data distribution, skewness, and kurtosis.
- Utilized descriptive statistics and visualizations.

## Models and Techniques
1. **Naive Bayesian:**
   - Utilized Gaussian Naive Bayes for classification.
   - Evaluated accuracy and performed cross-validation.

2. **K-Nearest Neighbors (KNN):**
   - Trained KNN classifier with different hyperparameters.
   - Assessed accuracy, confusion matrix, and classification report.

3. **Decision Tree (Entropy):**
   - Employed Decision Tree classifier using entropy criterion.
   - Evaluated accuracy and analyzed the model's performance.

4. **Neural Network (Multi-layer Perceptron):**
   - Standardized data for Neural Network.
   - Trained MLP classifier with hidden layers.
   - Assessed accuracy and created a confusion matrix.

5. **Bayesian Belief Network:**
   - Implemented a Bayesian Network using `pgmpy`.
   - Fitted the model and assessed its accuracy.

6. **PCA, LDA, SVD:**
   - Applied dimensionality reduction techniques such as PCA, LDA, and SVD.
   - Trained models with reduced features and evaluated accuracy.

## Evaluation and Metrics
- Utilized accuracy, precision, recall, F1-score, and error rate for model evaluation.
- Created ROC curves and analyzed area under the curve (AUC) for binary classification.

## Readme Suggestions
### How to Use
1. **Install Dependencies:**
   - Ensure required libraries are installed (`numpy`, `pandas`, `scikit-learn`, `pgmpy`, etc.).

2. **Run the Models:**
   - Execute the provided code in a Jupyter notebook or Python script.
   - Experiment with different models and parameters.

3. **Review Results:**
   - Analyze accuracy, precision, recall, and other metrics for each model.
   - Check visualizations, such as confusion matrices and ROC curves.

### Future Enhancements
- Implement additional models or ensemble methods.
- Explore hyperparameter tuning for existing models.
- Enhance data preprocessing and feature engineering.

### Contributors
- Mention Kaggle kernels or code snippets used for inspiration.
- Provide credit to sources or collaborators.

### License
This project is licensed under [MIT License](LICENSE). Feel free to use, modify, and distribute. Contributions are welcome!

Happy coding and happy spam detection!
