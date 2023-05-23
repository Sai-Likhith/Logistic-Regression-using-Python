# Logistic-Regression-using-Python

* Supervised Learning Model
* Primarily used for binary classification problems and Regression
* Linear regression + Sigmoid Function
Logistic regression is a statistical model used for binary classification problems. It is an extension of linear regression that predicts the probability of an input belonging to a specific class. Unlike linear regression, which predicts continuous values, logistic regression is designed to handle discrete outcomes.
 
The fundamental concept behind logistic regression is the logistic function, also known as the sigmoid function. The logistic function maps any real number to a value between 0 and 1. It takes the form:

sigmoid(z) = 1/((1 + exp(-z)))

![image](https://github.com/Sai-Likhith/Logistic-Regression-using-Python/assets/102646751/4fe09a20-1266-4f79-b666-aae46fab8696)


In logistic regression, the model applies this sigmoid function to a linear combination of the input features to obtain a value between 0 and 1. This value represents the estimated probability of the input belonging to a particular class.

To learn the parameters of the logistic regression model, it uses a technique called maximum likelihood estimation. The objective is to find the optimal set of coefficients that maximizes the likelihood of observing the labeled data. This process involves minimizing a cost function, often referred to as the cross-entropy loss, which measures the dissimilarity between the predicted probabilities and the true class labels.

Once the model is trained, it can make predictions on new, unseen data by calculating the probability of the input belonging to the positive class (class 1) based on the learned coefficients and feature values. By applying a chosen threshold (commonly 0.5), the model classifies the input into one of the two classes: positive or negative.

Logistic regression has several advantages. Firstly, it is a relatively simple and interpretable model. The coefficients can be easily interpreted as the influence of each feature on the probability of the outcome. This makes logistic regression useful for understanding the relationship between predictors and the response variable.

Additionally, logistic regression is computationally efficient and can handle large datasets. It requires fewer computational resources compared to more complex models like neural networks. Moreover, logistic regression provides probabilistic outputs, allowing for a better understanding of the uncertainty associated with each prediction.

However, logistic regression also has some limitations. It assumes a linear relationship between the input features and the log-odds of the outcome. If the relationship is non-linear, logistic regression may not capture it effectively. In such cases, feature engineering or more advanced techniques may be necessary.

Logistic regression is also sensitive to outliers. Outliers can disproportionately affect the estimated coefficients, leading to biased predictions. Thus, it is important to preprocess the data and handle outliers appropriately.

Logistic regression finds application in various domains. It is commonly used in areas such as spam detection, fraud detection, disease diagnosis, sentiment analysis, and churn prediction. Its simplicity, interpretability, and efficiency make it a popular choice when transparency and explainability are important.
# Advantages of Logistic Regression:

* Simplicity: Logistic regression is a relatively simple and interpretable model. It is easy to understand and implement, making it a good choice when transparency and explainability are important.
* Efficiency: Logistic regression can be trained efficiently even on large datasets. It has low computational requirements, making it computationally inexpensive compared to more complex models.
* Probability estimation: Logistic regression provides probabilistic outputs, allowing for a better understanding of the uncertainty associated with each prediction. This can be useful in decision-making processes.

# Limitations of Logistic Regression:
* Linearity assumption: Logistic regression assumes a linear relationship between the input features and the log-odds of the outcome. If the relationship is non-linear, logistic regression may not perform well and may require feature engineering or more advanced techniques.
* Limited complexity: Logistic regression is a linear model and cannot capture complex relationships or interactions between features as effectively as non-linear models like decision trees or neural networks.
* Sensitivity to outliers: Logistic regression can be sensitive to outliers, as it tries to minimize the overall error. Outliers can disproportionately affect the estimated coefficients and, consequently, the predictions.
# Applications of Logistic Regression:
* Spam Detection: Logistic regression can be used to identify spam emails by analyzing various features such as the email content, sender information, and subject line. It classifies emails as either spam or non-spam based on learned patterns from labeled training data.
* Fraud Detection: Logistic regression is utilized in fraud detection systems to identify fraudulent transactions or activities. By considering factors like transaction amounts, locations, and user behavior patterns, the model can predict the likelihood of a transaction being fraudulent.
* Disease Diagnosis: Logistic regression is employed in medical research and healthcare to predict the presence of certain diseases or conditions. By considering patient characteristics, symptoms, and diagnostic test results, the model can assist in diagnosing diseases such as cancer, diabetes, or heart disease.
* Sentiment Analysis: Logistic regression is used in sentiment analysis to determine the sentiment or opinion expressed in textual data. It can classify text as positive or negative based on the presence of certain words, sentiment indicators, or linguistic patterns. This application is valuable in social media monitoring, brand reputation management, and customer feedback analysis.
* Market Segmentation: Logistic regression is used in market research and customer segmentation to divide a population into distinct groups based on their characteristics, preferences, or behaviors. By analyzing demographic data, purchasing patterns, or survey responses, the model can identify segments with similar traits for targeted marketing strategies.
