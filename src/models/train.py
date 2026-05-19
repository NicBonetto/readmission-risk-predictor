import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_parquet('../../data/processed/features.parquet')
Y = pd.read_parquet('../../data/processed/labels.parquet')['readmitted_30d']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)

model = LogisticRegression(fit_intercept=True, class_weight='balanced', max_iter=1500, C=0.01)
model.fit(X_train, Y_train)

train_predictions = model.predict(X_train)

print(f"Training accuracy: {accuracy_score(Y_train, train_predictions)}")
print(f"Training recall: {recall_score(Y_train, train_predictions)}")
print(f"Training precision: {precision_score(Y_train, train_predictions)}")

test_predictions = model.predict(X_test)

print(f"Test accuracy: {accuracy_score(Y_test, test_predictions)}")
print(f"Test recall: {recall_score(Y_test, test_predictions)}")
print(f"Test precision: {precision_score(Y_test, test_predictions)}")

test_probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, test_probabilities)

plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Posititve Rate')
plt.show()

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values("coefficient")

coef_df.plot(kind="barh", x="feature", y="coefficient", figsize=(8, 6))
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
