import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# *Custom Naive Bayes Class*
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
        
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator

# *تحميل البيانات*
file_path = r'C:\Users\hp\Desktop\spam_assassin.csv'
data = pd.read_csv(file_path)

# أخذ عينة من البيانات لتقليل الحجم (اختياري)
data_sampled = data.sample(frac=0.5, random_state=42)  # استخدام 50% من البيانات
X = data_sampled['text']
y = data_sampled['target']

# تحويل النص إلى تمثيل عددي باستخدام TF-IDF مع تحديد عدد الميزات
vectorizer = TfidfVectorizer(max_features=5000)  # استخدام 5000 ميزة فقط
X_vectorized = vectorizer.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# تدريب النموذج
nb = NaiveBayes()
nb.fit(X_train.toarray(), y_train)
y_pred = nb.predict(X_test.toarray())

# تقييم النموذج
conf_matrix = confusion_matrix(y_test, y_pred)  # استخدم المتغير هنا
print(conf_matrix)
print(classification_report(y_test, y_pred))

# إحصائيات إضافية
tn, fp, fn, tp = conf_matrix.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = (fp + fn) / (tp + tn + fp + fn)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")

# تمثيل البيانات بالهيستوجرام
# تجميع البيانات حسب "target"
target_counts = data['target'].value_counts()

# رسم الهيستوجرام
plt.figure(figsize=(8, 5))
target_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Number of Emails by Target')
plt.xlabel('Target (Spam or Not Spam)')
plt.ylabel('Number of Emails')
plt.xticks(rotation=0)  # لتدوير تسميات المحور السيني
plt.show()
"""
precision    recall  f1-score   support

           0       0.82      0.99      0.90       387
           1       0.98      0.56      0.71       193

    accuracy                           0.85       580
   macro avg       0.90      0.78      0.81       580
weighted avg       0.87      0.85      0.84       580

Accuracy: 85.00%
Error Rate: 15.00%
"""
