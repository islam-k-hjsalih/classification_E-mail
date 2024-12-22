import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# تحميل الداتا
file_path = r'C:\Users\hp\Desktop\spam_assassin.csv'
data = pd.read_csv(file_path)

# طباعة البيانات والأعمدة
print(data)
print(data.columns)

# إعادة تسمية الأعمدة
data.columns = ['email', 'target']

# تقسيم البيانات إلى ميزات ومتغير الهدف
X = data['email']
y = data['target']

# تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحويل نصوص البريد الإلكتروني إلى بيانات عددية باستخدام CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# إنشاء مصنف KNN
knn = KNeighborsClassifier(n_neighbors=5)

# تدريب النموذج
knn.fit(X_train_vectorized, y_train)

# إجراء التنبؤات
y_pred = knn.predict(X_test_vectorized)

# تقييم النموذج
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)  
error_rate = (fp + fn) / (tp + tn + fp + fn)  

print(f"Accuracy :{accuracy * 100:.2f}%")
print(f"Error Rate : {error_rate * 100:.2f}%")

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
op/java oop programs/ISALM_KHALEEL_HJSALIH/src/knnshow.py"
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
