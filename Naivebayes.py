from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# *تحميل البيانات*
file_path = 'C:\Users\ac\Downloads\spam_assassin.csv.zip'  # استبدل بالمسار الصحيح لملف البيانات
data = pd.read_csv(file_path)

# *تحضير البيانات*
X = data['text']  # عمود النص
y = data['label']  # عمود التصنيف (Spam/Legitimate)

# *ترميز الفئات إذا كانت نصية*
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# *تحويل النص إلى تمثيل عددي باستخدام TF-IDF*
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# *تقسيم البيانات*
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# *تطبيق Naive Bayes*
nb_model = GaussianNB()
nb_model.fit(X_train.toarray(), y_train)  # تحويل إلى مصفوفة عادية
nb_predictions = nb_model.predict(X_test.toarray())

# *تقييم Naive Bayes*
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))


# إحصائيات إضافية
tn, fp, fn, tp = accuracy_score.ravel()

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
