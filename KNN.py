import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = r'C:\Users\hp\Desktop\spam_assassin.csv'
data = pd.read_csv(file_path)

# Display the first few rows and column names for exploration
print(data)
print(data.columns)

# Assuming the first column is 'email' and the second column is 'target'
# Rename columns for easier access
data.columns = ['email', 'target']

# Split the dataset into features and target variable
X = data['email']
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the email text into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors(k)

# Fit the model
knn.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = knn.predict(X_test_vectorized)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# حساب عدد التنبؤات الصحيحة والخاطئة
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# حساب نسبة الصواب والخطأ
accuracy = (tp + tn) / (tp + tn + fp + fn)  # نسبة الصواب
error_rate = (fp + fn) / (tp + tn + fp + fn)  # نسبة الخطأ

print(f"نسبة الصواب: {accuracy * 100:.2f}%")
print(f"نسبة الخطأ: {error_rate * 100:.2f}%")


"""


*************************###OUTPUT###**************************
                                                   text  target
0     From ilug-admin@linux.ie Mon Jul 29 11:28:02 2...       0
1     From gort44@excite.com Mon Jun 24 17:54:21 200...       1
2     From fork-admin@xent.com Mon Jul 29 11:39:57 2...       1
3     From dcm123@btamail.net.cn Mon Jun 24 17:49:23...       1
4     From ilug-admin@linux.ie Mon Aug 19 11:02:47 2...       0
...                                                 ...     ...
5791  From ilug-admin@linux.ie Mon Jul 22 18:12:45 2...       0
5792  From fork-admin@xent.com Mon Oct 7 20:37:02 20...       0
5793  Received: from hq.pro-ns.net (localhost [127.0...       1
5794  From razor-users-admin@lists.sourceforge.net T...       0
5795  From rssfeeds@jmason.org Mon Sep 30 13:44:10 2...       0

[5796 rows x 2 columns]
Index(['text', 'target'], dtype='object')
[[775   4]
 [ 19 362]]
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       779
           1       0.99      0.95      0.97       381

    accuracy                           0.98      1160
   macro avg       0.98      0.97      0.98      1160
weighted avg       0.98      0.98      0.98      1160

Accuracy :98.02%
Error Rate : 1.98%

"""
