# Spam-mail-prediction-project
# Introduction 
Spam mail is a significant problem in the digital age, causing inconvenience and posing security risks to users. Detecting and filtering out spam mail is crucial for maintaining a clean and secure inbox. This project aims to develop a machine learning model to predict whether a given email is spam (unwanted) or ham (legitimate) based on its content.

The project involves several key steps, from importing necessary libraries and preprocessing the data to training the model and making predictions on new emails. By leveraging techniques such as text feature extraction and logistic regression, we aim to build an effective spam detection system that can accurately classify emails.

# How It works 

Step 1:
Importing Libraries :  This step imports necessary libraries. numpy and pandas are used for data manipulation, train_test_split from sklearn is used to split the data into training and testing sets, TfidfVectorizer is used for transforming text data into numerical features, LogisticRegression is used for building the classification model, and accuracy_score is used to evaluate the model's performance.

Step 2: Data Collection and Preprocessing  :
This step reads the raw mail data from a CSV file and replaces any null values with empty strings. The head() function prints the first five rows of the dataframe to give an overview of the data.

Step 3: Label Encoding  :
This step converts the labels of the mail data from 'spam' and 'ham' to numerical values 0 and 1, respectively. This is known as label encoding, and it is necessary for the machine learning algorithm to process the labels.

Step 4: Separating the Data   :
This step separates the data into features (X) and labels (y). The text messages are stored in X, and the corresponding labels (spam or ham) are stored in y.


Step 5: Splitting the Data   :
This step splits the data into training and testing sets. 80% of the data is used for training, and 20% is used for testing. The random_state parameter ensures reproducibility of the split.


Step 6: Feature Extraction  :
This step converts the text data into numerical features using the TfidfVectorizer. It transforms the text into a matrix of TF-IDF features, which are used for training the model. The fit_transform method is applied to the training data, and the transform method is applied to the test data.



Step 7: Convert Labels to Integers   :
This step converts the labels Y_train and Y_test to integers. This ensures that the labels are in a suitable format for the logistic regression model.


Step 8: Training the Model   :
This step initializes the logistic regression model and trains it using the training data features and labels.


Step 9: Evaluating the Trained Model   :
This step evaluates the trained model by making predictions on the training and test data. The accuracy scores for both sets are calculated to assess the model's performance.


Step 10: Making a Prediction   :
This step makes a prediction on a new input mail. The mail is transformed into numerical features using the same TfidfVectorizer used for training. The model then predicts whether the mail is spam (0) or ham (1). The result is printed, indicating whether the mail is spam or not.






