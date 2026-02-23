📊 Customer Churn Prediction using PySpark (MLlib)


🚀 Project Overview

This project builds a Customer Churn Prediction model using PySpark MLlib on the Bank Churn Modelling dataset.
The objective is to predict whether a customer will churn (1) or not churn (0) using demographic and financial features.
Unlike typical small-scale sklearn projects, this implementation uses Apache Spark, making it scalable for large datasets and distributed environments.

🎯 Problem Statement
Banks face significant revenue loss due to customer churn.
The goal of this project was to:
Analyze customer-level banking data
Engineer features using Spark ML
Build classification models
Compare performance using accuracy metric

🛠 Tech Stack

Python
PySpark
Spark MLlib
Jupyter Notebook

📂 Dataset Used

File: Bank Churn Modelling.csv

The dataset includes features such as:
CreditScore
Geography
Gender
Tenure
Balance
Num Of Products
Has Credit Card
Is Active Member
Estimated Salary
Churn (Target Variable)

⚙️ Implementation Workflow
1️⃣ Spark Session Initialization
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

Loaded dataset using:

spark.read.csv(..., header=True, inferSchema=True)
2️⃣ Data Cleaning

Dropped null values using:

sdf = sdf.dropna()
3️⃣ Feature Engineering
🔹 Categorical Encoding

Used StringIndexer to encode:
Geography → iGeography
Gender → iGender
StringIndexer(inputCols=['Geography','Gender'], 
              outputCols=['iGeography','iGender'])
🔹 Feature Vector Creation

Used VectorAssembler to combine features into a single feature vector X.
Features included:

CreditScore
Tenure
Balance
Num Of Products
Has Credit Card
Is Active Member
Estimated Salary
iGeography
iGender
VectorAssembler(..., outputCol='X')


4️⃣ Train-Test Split
train, test = data.randomSplit([0.7, 0.3], seed=2529)
70% Training
30% Testing

5️⃣ Feature Scaling
Applied StandardScaler for scaled logistic regression model:
StandardScaler(inputCol='X', outputCol='sX', withStd=True, withMean=True)
🤖 Models Implemented

1️⃣ Logistic Regression (Scaled Features)
LogisticRegression(featuresCol='sX', labelCol='Churn')
Used scaled features (sX) for better optimization performance.

2️⃣ Logistic Regression (Unscaled)
LogisticRegression(featuresCol='X', labelCol='Churn')
Used original feature vector without scaling.

3️⃣ Random Forest Classifier
RandomForestClassifier(featuresCol='X', labelCol='Churn')
Tree-based ensemble model for improved non-linear classification.

📊 Model Evaluation

Used:

MulticlassClassificationEvaluator(
    predictionCol='prediction',
    labelCol='Churn',
    metricName='accuracy'
)

Evaluation Metric:

✅ Accuracy

Each model’s accuracy was calculated on the test dataset for performance comparison.
🔍 Key Learning Outcomes

Implemented distributed ML using PySpark
Understood Spark DataFrame transformations
Applied feature engineering using MLlib pipeline components
Compared linear and ensemble classification models
Learned scaling importance in Logistic Regression

📦 Project Structure
📦 Customer-Churn-PySpark
 ┣ 📜 ChurnMODEL_Samar.ipynb
 ┣ 📜 Bank Churn Modelling.csv
 ┣ 📜 README.md
▶️ How to Run

Install PySpark
pip install pyspark
Open Jupyter Notebook
jupyter notebook

Run all cells sequentially

💼 Business Relevance

This model can help banks:
Identify high-risk customers early
Enable targeted retention campaigns
Improve customer lifetime value
Reduce churn-related revenue losses
Because it is built on Spark, the solution is scalable for enterprise-level banking datasets.

👨‍💻 Author

Samar Singh Rathore
