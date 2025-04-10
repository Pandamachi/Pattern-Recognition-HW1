import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# 1. Dry Bean Dataset
df = pd.read_csv("Dry_Bean_DataSet.csv")

# 2. Oversampling
max_count = df["Class"].value_counts().max()
df_oversampled = pd.DataFrame()

for label in df["Class"].unique():
    subset = df[df["Class"] == label]
    upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    df_oversampled = pd.concat([df_oversampled, upsampled])

df = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)  # 打亂

# 3. Data Preprocessing
X = df_oversampled.drop(columns=["Class"])
y = df_oversampled["Class"]

X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

# 4. Naive Bayes
class GaussianNaiveBayesMulti:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.stds = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.stds[c] = X_c.std(axis=0) + 1e-6
            self.priors[c] = len(X_c) / len(X)

    def _likelihood(self, mean, std, x):
        exponent = np.exp(-0.5 * ((x - mean) / std) ** 2)
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = self._likelihood(self.means[c], self.stds[c], x)
            likelihood = np.clip(likelihood, 1e-9, None)  # 避免 log(0)
            cond = np.sum(np.log(likelihood))
            posteriors.append(prior + cond)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._posterior(x) for x in X])

nb_model = GaussianNaiveBayesMulti()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

# 5. Logistic Regression（Softmax）
class LogisticRegressionScratchMulti:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def _softmax(self, z):
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.k = len(np.unique(y))
        self.weights = np.zeros((self.n, self.k))
        self.bias = np.zeros(self.k)

        y_onehot = np.eye(self.k)[y]

        for _ in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            probs = self._softmax(logits)
            error = probs - y_onehot

            dw = np.dot(X.T, error) / self.m
            db = np.sum(error, axis=0) / self.m

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self._softmax(logits)

logreg_model = LogisticRegressionScratchMulti()
logreg_model.fit(X_train_scaled, y_train)
y_pred_lr = logreg_model.predict(X_test_scaled)

# 6. Confusion Matrix
def plot_conf_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}_over.png")
    plt.show()

class_labels = le.classes_.astype(str)
plot_conf_matrix(y_test, y_pred_nb, class_labels, "Naive Bayes Confusion Matrix")
plot_conf_matrix(y_test, y_pred_lr, class_labels, "Logistic Regression Confusion Matrix")

print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb, target_names=class_labels))

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=class_labels))

# 7. Feature Importance
def plot_all_class_feature_importance(weights, feature_names, class_labels):
    num_classes = weights.shape[1]
    fig, axes = plt.subplots(1, num_classes, figsize=(20, 6), sharey=True)

    for i in range(num_classes):
        importance = weights[:, i]
        sorted_idx = np.argsort(np.abs(importance))[::-1]

        sns.barplot(
            x=importance[sorted_idx],
            y=np.array(feature_names)[sorted_idx],
            ax=axes[i],
            palette="coolwarm"
        )
        axes[i].set_title(f"'{class_labels[i]}'")
        axes[i].set_xlabel("Weight")
        if i == 0:
            axes[i].set_ylabel("Feature")
        else:
            axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig("feature_importance_over.png")
    plt.show()

plot_all_class_feature_importance(logreg_model.weights, X.columns, class_labels)

from sklearn.metrics import accuracy_score, f1_score

def print_summary_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    support = len(y_true)

    print(f"\n=== {model_name} Summary ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f} (macro average)")
    print(f"Support  : {support}")

print_summary_metrics(y_test, y_pred_nb, "Naive Bayes")
print_summary_metrics(y_test, y_pred_lr, "Logistic Regression")
