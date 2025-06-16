import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from src.train_model import build_siamese_lstm
from src.utils.tokenization import load_tokenizer, texts_to_padded_sequences
from sklearn.metrics import f1_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NUM_WORDS = 50000
EMBEDDING_DIM = 100
MAXLEN = 40

data_path = BASE_DIR / "data" / "train_pairs.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

s1 = [d["sentence1"] for d in data]
s2 = [d["sentence2"] for d in data]
y_all = np.array([d["label"] for d in data], dtype=int)

idx = np.arange(len(y_all))
np.random.seed(42)
np.random.shuffle(idx)
split_at = int(0.9 * len(idx))
val_idx = idx[split_at:]

s1_val = [s1[i] for i in val_idx]
s2_val = [s2[i] for i in val_idx]
y_val = y_all[val_idx]

model = build_siamese_lstm(
    vocab_size=NUM_WORDS,
    embedding_dim=EMBEDDING_DIM,
    maxlen=MAXLEN
)
model.load_weights(str(BASE_DIR / "models" / "lstm_model_batch128.h5"))

tokenizer = load_tokenizer(str(BASE_DIR / "models" / "tokenizer_batch128.pkl"))
MAXLEN = model.input_shape[0][1]

X1_val = texts_to_padded_sequences(tokenizer, s1_val, MAXLEN)
X2_val = texts_to_padded_sequences(tokenizer, s2_val, MAXLEN)
y_prob = model.predict([X1_val, X2_val], batch_size=256).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# Classification report
print("Classification Report:\n")
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_val, y_prob)
plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Training history plot (если файл history.json есть)
hist_path = BASE_DIR / "models" / "history.json"
if hist_path.exists():
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    # Plot loss
    plt.figure()
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    # Plot accuracy
    plt.figure()
    plt.plot(history['accuracy'], label='train_acc')
    plt.plot(history['val_accuracy'], label='val_acc')
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
else:
    print("No history.json found — skip training curves.")

best_thr, best_f1 = 0.0, 0.0
for thr in np.linspace(0.1, 0.9, 81):
    y_pred_thr = (y_prob >= thr).astype(int)
    f1 = f1_score(y_val, y_pred_thr)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"Best F1={best_f1:.4f} at threshold={best_thr:.2f}")