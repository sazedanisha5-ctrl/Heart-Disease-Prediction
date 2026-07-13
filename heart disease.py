import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/sazed/OneDrive/Documents/heart.csv")
data = pd.get_dummies(data, drop_first=True)

print(data.head())

X = data.drop("target", axis=1)
y = data["target"]

from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

test_sizes = [0.2, 0.3, 0.4, 0.5]

print("---- Split Accuracy Results ----")

for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Test Size {int(size*100)}% -> Accuracy: {acc:.2f}")

from sklearn.model_selection import train_test_split

# Example dataset
X = data.drop("target", axis=1)
y = data["target"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import matplotlib.pyplot as plt

test_sizes = [20, 30, 40, 50]
accuracy = [99, 99, 98, 97]

plt.figure()

plt.bar(test_sizes, accuracy)

plt.xlabel("Test Size (%)")
plt.ylabel("Accuracy (%)")
plt.title("Train-Test Split vs Accuracy")

plt.show()
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create model
svm_model = SVC(kernel='linear')

# Train model
svm_model.fit(X_train, y_train)

# Predict
svm_pred = svm_model.predict(X_test)

# Accuracy
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Precision:", precision_score(y_test, svm_pred))
print("SVM Recall:", recall_score(y_test, svm_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Print results
print("\n----- RANDOM FOREST -----")
print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print("RF Precision:", precision_score(y_test, rf_pred))
print("RF Recall:", recall_score(y_test, rf_pred))

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Scale data
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train
knn_model.fit(X_train_sc, y_train)

# Predict
knn_pred = knn_model.predict(X_test_sc)

# Metrics
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Precision:", precision_score(y_test, knn_pred))
print("KNN Recall:", recall_score(y_test, knn_pred))

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create model
lgb_model = lgb.LGBMClassifier()

# Train
lgb_model.fit(X_train_sc, y_train)

# Predict
lgb_pred = lgb_model.predict(X_test_sc)

# Metrics
print("\n----- LIGHTGBM -----")
print("LGBM Accuracy:", accuracy_score(y_test, lgb_pred))
print("LGBM Precision:", precision_score(y_test, lgb_pred))
print("LGBM Recall:", recall_score(y_test, lgb_pred))


import matplotlib.pyplot as plt

plt.figure()
plt.hist(svm_pred)
plt.title("Histogram of SVM Predictions")
plt.xlabel("Predicted Class (0 = No Disease, 1 = Disease)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(rf_pred)
plt.title("Histogram of Random Forest Predictions")
plt.xlabel("Predicted Class (0 = No Disease, 1 = Disease)")
plt.ylabel("Frequency")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Models
models = ["SVM", "RF", "LGBM", "KNN"]

# Metrics (in %)
accuracy = [80.49, 98.54, 98.05, 89.76]
precision = [76.92, 100, 99.01, 88.68]
recall = [87.38, 97.08, 97.08, 91.26]

x = np.arange(len(models))
width = 0.25

plt.figure()

# Colored bars
plt.bar(x - width, accuracy, width, label="Accuracy", color="#4CAF50")   # Green
plt.bar(x, precision, width, label="Precision", color="#2196F3")        # Blue
plt.bar(x + width, recall, width, label="Recall", color="#FF9800")      # Orange

# Labels
plt.xlabel("Models")
plt.ylabel("Percentage (%)")
plt.title("Model Comparison (Accuracy, Precision, Recall)")
plt.xticks(x, models)

# Add values on top
for i in range(len(models)):
    plt.text(x[i] - width, accuracy[i], str(accuracy[i]), ha='center')
    plt.text(x[i], precision[i], str(precision[i]), ha='center')
    plt.text(x[i] + width, recall[i], str(recall[i]), ha='center')

plt.legend()

plt.show()

import matplotlib.pyplot as plt

# y = target column (0 or 1)

plt.figure()
plt.hist(y)

plt.title("Heart Disease Dataset Distribution (1025 Samples)")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")

plt.show()

import numpy as np

counts = np.bincount(y)

plt.figure()
plt.hist(y)

for i, count in enumerate(counts):
    plt.text(i, count, str(count))

plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()

plt.figure()
plt.bar(['No Disease', 'Disease'], counts)

plt.title("Heart Disease Dataset (1025 Samples)")
plt.ylabel("Number of Patients")

plt.show()

plt.savefig("graph.png", dpi=300, bbox_inches='tight')

# ===== IMPORTS =====
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ===== CONFUSION MATRICES =====
cm_svm = confusion_matrix(y_test, svm_pred)
cm_rf = confusion_matrix(y_test, rf_pred)
cm_lgbm = confusion_matrix(y_test, lgb_pred)
cm_knn = confusion_matrix(y_test, knn_pred)

# ===== FUNCTION TO PLOT =====
def plot_cm(cm, title):
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Add numbers inside boxes
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha='center', va='center')

# ===== PLOT ALL =====
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plot_cm(cm_svm, "SVM")

plt.subplot(2,2,2)
plot_cm(cm_rf, "Random Forest")

plt.subplot(2,2,3)
plot_cm(cm_lgbm, "LightGBM")

plt.subplot(2,2,4)
plot_cm(cm_knn, "KNN")

plt.tight_layout()
plt.show()
