# Fraud Detection System for Online Transactions

## Overview

This project implements a **Fraud Detection System** for online transactions using machine learning. The model leverages a **Decision Tree Classifier** to predict fraudulent transactions based on transaction details such as transaction type, amount, and balance before and after the transaction.

The system is designed to detect fraudulent transactions by identifying anomalies in transaction data. It has an accuracy of **99.67%**, demonstrating its effectiveness in identifying fraudulent activity in financial transactions.

## Table of Contents
- [Objective](#objective)
- [Business Objective](#business-objective)
- [How It Works](#how-it-works)
- [Model](#model)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)
- [Live Demo](#live-demo)

## Objective

The primary goal of this system is to predict fraudulent transactions and minimize financial losses for businesses and customers. By using machine learning, the model helps identify transactions that are likely to be fraudulent, based on transaction features. The system is an essential tool for fraud detection teams, enabling them to take immediate actions to prevent financial crimes.

## Business Objective

Financial institutions and e-commerce platforms face significant losses due to fraudulent activities. The business objective of this system is to provide an automated solution for detecting fraudulent transactions, ensuring that businesses can act swiftly to mitigate losses. With its high accuracy rate, the model helps businesses protect their customers, reduce financial risks, and improve trust in their services.

## How It Works

1. **Data Collection**: Transaction data is collected, including details such as transaction type (e.g., CASH_OUT, PAYMENT), transaction amount, and the balances before and after the transaction.
   
2. **Data Preprocessing**: The data is cleaned and preprocessed, with categorical variables being encoded for use in the model.

3. **Model Training**: A **Decision Tree Classifier** is trained on labeled data (fraudulent and non-fraudulent transactions) to learn patterns associated with fraudulent activity.

4. **Prediction**: The trained model is deployed to make real-time predictions. It receives new transaction data as input and outputs a prediction indicating whether the transaction is fraudulent or not.

## Model

The **Decision Tree Classifier** was selected for this task due to its simplicity and interpretability, which makes it an excellent choice for financial fraud detection. The model is trained on features such as:
- Transaction Type (e.g., CASH_OUT, PAYMENT, etc.)
- Transaction Amount
- Original Balance Before the Transaction
- New Balance After the Transaction

The model achieved a high accuracy of **99.67%** in detecting fraudulent transactions, based on evaluation metrics like the **Confusion Matrix** and **Classification Report**.

![Screenshot 2024-11-07 001739](https://github.com/user-attachments/assets/712ef150-3445-4825-ba36-8e8782540911)



![Screenshot 2024-11-07 001730](https://github.com/user-attachments/assets/2cd15e52-fb35-49c7-91be-53f8ee062001)



## Future Improvements

While the current model provides reliable predictions, there are several areas where improvements can be made:
1. **Model Tuning**: Exploring other algorithms such as Random Forest, SVM, or XGBoost to further enhance prediction accuracy.
2. **Data Augmentation**: Using more diverse and comprehensive datasets, including additional features (e.g., transaction time, customer history) for better model performance.
3. **Real-time Detection**: Implementing real-time fraud detection for live transactions, integrating the system with financial services.
4. **Model Explainability**: Using tools like SHAP or LIME to explain model decisions and make the system more transparent for end-users.

## Technologies Used

- **Python**: The programming language used for data processing, model training, and development.
- **Scikit-learn**: The library used for training the Decision Tree Classifier.
- **Streamlit**: The framework for building the user interface and deploying the web application.
- **Pandas and NumPy**: For data manipulation and processing.
- **Matplotlib and Seaborn**: For data visualization and evaluating model performance.

## How to Use

### Prerequisites
1. Install required libraries:
2. Clone the repository: git clone https://github.com/SUMIT2001GO/Fraud-Detection-System-Using-Online-Transaction-.git
3. Save the trained model (`fraud_detection_model.joblib`) and the app file (`app.py`) in the same directory.

### Running the Application
To start the application, navigate to the project folder and run: app.py

### Input
Once the app is running, enter the following transaction details to get a prediction:
- **Transaction Type**: Select from `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`.
- **Transaction Amount**: Enter the transaction amount.
- **Original Balance Before Transaction**: Enter the balance before the transaction.
- **New Balance After Transaction**: Enter the balance after the transaction.

Click on the "Predict Fraud" button to get the prediction result. The model will predict whether the transaction is a fraud or not.

## Live Demo

You can try the model online using the following link:  
[Live Demo: Fraud Detection System](https://fraud-detection-system-online-transaction.streamlit.app/)


![Screenshot 2024-11-07 001934](https://github.com/user-attachments/assets/5ec029c2-db72-4bdb-bd50-02f29f7eccab)


![Screenshot 2024-11-07 002027](https://github.com/user-attachments/assets/3af2c37f-3691-491f-a513-b1c7ccafa1fe)



## Acknowledgements

We would like to express our gratitude to the following for their contributions:

- **Scikit-learn**: For providing an easy-to-use library for building and training machine learning models.
- **Streamlit**: For enabling quick development and deployment of the web application.
- **Open Source Community**: For their invaluable resources, libraries, and tutorials that made it possible to develop this system.


## Conclusion

This fraud detection system offers an effective, high-accuracy tool for identifying fraudulent transactions in online platforms. By leveraging machine learning, businesses can prevent fraudulent activities, saving money and improving customer trust. Further model improvements can lead to even better performance and real-time fraud detection capabilities.

