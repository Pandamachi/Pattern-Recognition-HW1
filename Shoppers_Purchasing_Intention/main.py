import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from sklearn.model_selection import train_test_split

# ===== 1. data preprocessing =====
df = pd.read_csv("data.csv")

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

# # Over-sampling
# revenue_counts = df['Revenue'].value_counts()
# print("Revenue 分布：")
# print(revenue_counts)

# majority_class = df[df['Revenue'] == revenue_counts.idxmax()]
# minority_class = df[df['Revenue'] == revenue_counts.idxmin()]
# n_to_sample = len(majority_class) - len(minority_class)

# minority_oversampled = minority_class.sample(n=n_to_sample, replace=True, random_state=42)
# df_balanced = pd.concat([df, minority_oversampled], ignore_index=True).sample(frac=1, random_state=42)


X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values
feature_names = df.drop("Revenue", axis=1).columns.tolist()

X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 2. Naive Bayes =====
class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                cond = np.sum(np.log(self._pdf(c, x)))
                class_probs.append(prior + cond)
            probs.append(np.exp(class_probs) / np.sum(np.exp(class_probs)))
        return np.array(probs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ===== 3. Logistic Regression =====
class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.W = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iter):
            linear = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(linear)
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)
            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ===== 4. Confusion Matrix =====
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
    plt.close()

# ===== 5. evulation module =====
def evaluate_and_save(name, model, X_val, y_val, probs=None):
    y_pred = model.predict(X_val)
    if probs is None:
        probs = model.predict_proba(X_val)

    cm_title = f"{name} - Confusion_Matrix"
    plot_confusion_matrix(y_val, y_pred, cm_title)
    cm_path = f"{cm_title}.png"

    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    roc_path = f"{name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    support_0 = sum(y_val == 0)
    support_1 = sum(y_val == 1)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Class 0", "Class 1"]))

    return acc, roc_auc, f1, (support_0, support_1), cm_path, roc_path

# ===== 6. output =====
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_acc, nb_auc, nb_f1, nb_support, nb_cm_path, nb_roc_path = evaluate_and_save(
    "NaiveBayes", nb_model, X_val, y_val, nb_model.predict_proba(X_val)[:, 1]
)

lr_model = LogisticRegression(lr=0.1, n_iter=1000)
lr_model.fit(X_train, y_train)
lr_acc, lr_auc, lr_f1, lr_support, lr_cm_path, lr_roc_path = evaluate_and_save(
    "LogisticRegression", lr_model, X_val, y_val
)

# ===== 7. Feature Importance =====
abs_weights = np.abs(lr_model.W)
sorted_idx = np.argsort(abs_weights)[::-1]
top_features = np.array(feature_names)[sorted_idx]
top_weights = abs_weights[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(top_features[:10][::-1], top_weights[:10][::-1], color='skyblue')
plt.xlabel("Absolute Weight")
plt.title("Top 10 Influential Features (Logistic Regression)")
plt.tight_layout()
plt.savefig("feature_importance_logistic_regression.png")
plt.close()

most_influential_feature = top_features[0]
print("\n===== Summary =====")
print(f"Naive Bayes - Accuracy: {nb_acc:.4f}, AUC: {nb_auc:.4f}, F1: {nb_f1:.4f}, Support: {nb_support}")
print(f"Logistic Reg - Accuracy: {lr_acc:.4f}, AUC: {lr_auc:.4f}, F1: {lr_f1:.4f}, Support: {lr_support}")
print(f"Most Influential Feature: {most_influential_feature}")

def plot_combined_roc(y_val, nb_probs, lr_probs):
    fpr_nb, tpr_nb, _ = roc_curve(y_val, nb_probs)
    auc_nb = auc(fpr_nb, tpr_nb)

    fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.2f})")
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_roc_curve.png")
    plt.close()

plot_combined_roc(
    y_val,
    nb_model.predict_proba(X_val)[:, 1],
    lr_model.predict_proba(X_val)
)