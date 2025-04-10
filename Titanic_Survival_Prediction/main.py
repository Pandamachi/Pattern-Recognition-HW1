import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score

# === 1. Data Preprocessing ===
df = pd.read_csv("train_cleaned.csv")
df = df.drop(columns=["PassengerId"])

df_majority = df[df.Survived == 0]
df_minority = df[df.Survived == 1]

# df_minority_upsampled = resample(
#     df_minority,
#     replace=True,
#     n_samples=len(df_majority),
#     random_state=42
# )

# df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# === 2. Naive Bayes ===
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.stds = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.stds[c] = X_c.std(axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _calculate_likelihood(self, mean, std, x):
        exponent = np.exp(-0.5 * ((x - mean) / std) ** 2)
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._calculate_likelihood(self.means[c], self.stds[c], x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X])


nb_model = GaussianNaiveBayes()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)


# === 3. Logistic Regression ===
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


logreg_model = LogisticRegressionScratch(lr=0.1, epochs=1000)
logreg_model.fit(X_train_scaled, y_train)
y_pred_lr = logreg_model.predict(X_test_scaled)
y_proba_lr = logreg_model.predict_proba(X_test_scaled)
y_proba_nb = [1 if p == 1 else 0 for p in y_pred_nb]  # Naive Bayes is hard prediction only


# === 4. Evaluation module ===
# Confusion Matrix & Classification Report
print("=== Naive Bayes ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

print("\n=== Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# ROC Curve
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
auc_nb = auc(fpr_nb, tpr_nb)
auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.2f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")


# Feature Importance for Logistic Regression
def plot_feature_importance(weights, feature_names, title):
    importance = np.abs(weights)
    sorted_idx = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Absolute Weight (Importance)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

feature_names = X.columns
plot_feature_importance(logreg_model.weights, feature_names, "Logistic Regression Feature Importance")

def print_metrics(y_true, y_pred, y_proba, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = auc(*roc_curve(y_true, y_proba)[:2])
    support = len(y_true)

    print(f"\n=== {model_name} Performance Summary ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc_score:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Support  : {support}")

print_metrics(y_test, y_pred_nb, y_proba_nb, "Naive Bayes")
print_metrics(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")