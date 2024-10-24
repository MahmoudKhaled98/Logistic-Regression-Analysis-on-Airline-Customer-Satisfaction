# Logistic Regression Analysis on Airline Customer Satisfaction

## **Project Overview**

In this project, I aim to explore whether in-flight entertainment experiences influence customer satisfaction through a binomial logistic regression model. Understanding this relationship can guide the airline in enhancing service features to improve satisfaction and future business outcomes. The dataset includes various flight-related factors from 129,880 customers, such as flight class, distance, and in-flight entertainment ratings.

The goal is to predict customer satisfaction based on their feedback using logistic regression, perform data exploration and cleaning, and evaluate the model's performance.

## **Table of Contents**

- [Introduction](#introduction)
- [Step 1: Imports](#step-1-imports)
- [Step 2: Data Exploration, Cleaning, and Preparation](#step-2-data-exploration-cleaning-and-preparation)
- [Step 3: Model Building](#step-3-model-building)
- [Step 4: Results and Evaluation](#step-4-results-and-evaluation)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [References](#references)

## **Introduction**

This analysis focuses on predicting customer satisfaction for an airline, using logistic regression to evaluate the impact of in-flight entertainment on customer experience. The goal is to understand whether enhancing the in-flight entertainment experience would increase customer satisfaction, helping the airline prioritize their investments in service improvements.

## Step 1: Imports


### Import Packages
To begin, I imported the necessary libraries for data manipulation, visualization, and model building:
- `pandas` for data handling
- `seaborn` and `matplotlib` for visualizations
- `sklearn` for splitting the data and model training (`train_test_split`, `LogisticRegression`, and `sklearn.metrics` for evaluation)

### Load the Dataset
The data consists of customer feedback on various aspects of their flight experience, including satisfaction levels.



## Step 2: Data Exploration, Cleaning, and Preparation

**Exploring the Data** 

The dataset includes multiple features such as flight class, distance, and in-flight entertainment ratings. After examining the dataset, the key features include:

- Flight Distance
- In-flight Entertainment
- Class
- Satisfaction

**Checking for Missing Values**

A quick data check revealed missing values in the "Arrival Delay in Minutes" column. Since this accounts for only 0.3% of the dataset, these rows were removed.

**Encoding Categorical Variables**


The satisfaction variable was encoded numerically (1 = satisfied, 0 = unsatisfied) for use in the logistic regression model.

**Creating Training and Testing Sets**

`70%` of the data was used for training, and the remaining `30%` was reserved for testing.


## Step 3: Model Building

**Fitting the Logistic Regression Model**

Using LogisticRegression from `sklearn`, the model was trained on the training data.


## Step 4: Results and Evaluation
**Model Coefficients**

- `coef = 0.99752883`
- `odds ratios = 2.71157278`
- `percentage change = 1.71 %`
- `probability = 73.06 %`

**Log-odds interpretation:**
For every `one-unit` increase in the `in-flight entertainment` rating, the log(odds) of being `satisfied` increase by `0.998`.

**Odds interpretation:**
For every `one-unit` increase in the `in-flight entertainment` rating, the odds of being `satisfied` increase by a factor of `2.712`.

**Percentage change in odds:**
This corresponds to a `171%` increase in the odds of being `satisfied` for each `one-unit` increase in the `in-flight entertainment` rating.

**Probability interpretation:**
When the `in-flight entertainment` rating increases by `one unit`, the `probability` of the user being `satisfied` is approximately `73.06%`.


**Predicting Test Data**

The model's predictions were evaluated on the test dataset, and the predicted labels were stored.

**Model Performance and Metrics**

- **`Accuracy: 0.8015`**
   - Interpretation: `80.15%` of all predictions (both satisfied and unsatisfied) are correct. However, accuracy alone can be misleading when the data is imbalanced.

- **`Precision: 0.8161`**
   - Interpretation: Of all the instances predicted as satisfied, `81.61%` were actually satisfied. Precision focuses on minimizing false positives (cases wrongly predicted as satisfied).

- **`Recall: 0.8215`**
   - Interpretation: Of all the actual satisfied cases, `82.15%` were correctly identified by the model. Recall focuses on minimizing false negatives (cases that are actually satisfied but predicted as unsatisfied).
   
- **`F1-Score: 0.8188`**
   - Interpretation: `The F1-score` is a balance between precision and recall. Here, it’s `81.88%`, indicating a good balance between precision and recall.
   
These metrics suggest a strong balance between precision and recall, meaning the model is effective in both minimizing false positives and identifying satisfied customers.


**Confusion Matrix**

The confusion matrix reveals the types of prediction errors made by the model:

- True Positives (TP): 17,423 customers correctly predicted as satisfied.

- True Negatives (TN): 13,714 customers correctly predicted as unsatisfied.

- False Negatives (FN): 3,925 customers incorrectly predicted as unsatisfied but were satisfied.

- False Positives (FP): 3,785 customers incorrectly predicted as satisfied but were unsatisfied.

## **Conclusion**


**Key Insights**

*   **Satisfaction Prediction:** The model predicts customer satisfaction with `80.2%` `accuracy`.

*   **In-flight Entertainment Impact:** When the inflight entertainment rating increases by one unit, the `probability` of the user being `satisfied` is approximately `73.06%`.


* **Logistic Regression Performance:** The model performs well in predicting customer satisfaction, suggesting that `in-flight entertainment` is a `significant factor`.

**Summary for Stakeholders**

* **In-flight Entertainment Focus:** 
Customers who rated in-flight entertainment highly were more likely to be satisfied. Improving in-flight entertainment should lead to better customer satisfaction, Given the strong relationship between in-flight entertainment and customer satisfaction, investments in improving this service feature should be prioritized to increase overall satisfaction.


* **Further Model Development:** Adding other variables (e.g., flight distance, departure delays) could potentially improve model accuracy and provide deeper insights into other satisfaction drivers.

## How to Run

1. **Clone the repository**:

    ```bash
    git clone <https://github.com/MahmoudKhaled98/Logistic-Regression-Analysis-on-Airline-Customer-Satisfaction.git>
    ```

2. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter notebook**:

    ```bash
    jupyter notebook
    ```
## References

- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
