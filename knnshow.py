import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# تحميل الداتا
file_path = r'C:\Users\hp\Desktop\spam_assassin.csv'
data = pd.read_csv(file_path)

# استخدام عينة من البيانات
data_sample = data.sample(frac=0.1, random_state=42)  # استخدام 10% من البيانات
data_sample.columns = ['email', 'target']

# Split the dataset into features and target variable
X = data_sample['email']
y = data_sample['target']

# Convert the email text into numerical data using CountVectorizer
vectorizer = CountVectorizer(max_features=5)  # استخدام 1000 ميزة فقط
X_vectorized = vectorizer.fit_transform(X)

# استخدام TruncatedSVD لتقليل الأبعاد إلى 2
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_vectorized)

# رسم البيانات
plt.figure(figsize=(10, 6))
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('SVD of Email Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Target (0: Not Spam, 1: Spam)')
plt.show()