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