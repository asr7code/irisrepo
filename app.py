# Colab-ready: train, save, and download an Iris model (scaler + LogisticRegression)
# Run this single cell in Google Colab. It will train, save `iris_model.pkl` and download it.

# Step 0: installs (Colab usually has these but safe to ensure)
!pip install -q scikit-learn pandas joblib

# Step 1: Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# (Colab-specific) utility to download files
try:
    from google.colab import files
    _can_download = True
except Exception:
    _can_download = False

# ---------- Step 2: Load the dataset ----------
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# ---------- Step 3: Basic EDA (printed) ----------
print("🔍 First 5 rows of dataset:")
display(df.head())

print("\n📊 Class distribution:")
print(df['species'].value_counts())

# optional visualization (uncomment if you want)
# sns.pairplot(df, hue='species')
# plt.show()

# ---------- Step 4: Train/Test split ----------
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Step 5: Build pipeline and train ----------
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, multi_class='auto', solver='lbfgs', random_state=42)
)

pipeline.fit(X_train, y_train)

# ---------- Step 6: Evaluation ----------
y_pred = pipeline.predict(X_test)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy Score:", acc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# optional: show confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

# ---------- Step 7: Save artifact (pipeline + metadata) ----------
artifact = {
    "pipeline": pipeline,                 # scaler + model
    "feature_names": iris.feature_names,  # list-like of feature names
    "target_names": list(iris.target_names),
    "sklearn_version": __import__("sklearn").__version__
}

OUT_PATH = Path("iris_model.pkl")
joblib.dump(artifact, OUT_PATH)
print(f"\nSaved model artifact to: {OUT_PATH.resolve()}")

# ---------- Step 8: Trigger download in Colab (if available) ----------
if _can_download:
    print("Preparing download...")
    files.download(str(OUT_PATH))
    print("Download started — check your browser. If it doesn't start, download the file from the Colab file browser.")
else:
    print("Not running in Colab environment or google.colab.files not available.")
    print("You can still download iris_model.pkl from the notebook environment (Files sidebar) or copy it to Drive.")
.after running and downloading , now i will create a irisrepo repository . You give me app.py and requirements.txt
