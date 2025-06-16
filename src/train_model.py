import json
from pathlib import Path
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dropout, Dense,
    Lambda, Multiply, Concatenate
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, Callback
)
from tensorflow.keras.optimizers import Adam

from src.utils.tokenization import (
    build_tokenizer,
    save_tokenizer,
    texts_to_padded_sequences
)


class DynamicDropoutCallback(Callback):
    """
    Callback, который изменяет Dropout.rate на каждой эпохе.
    epoch == 0 → rate = 0.5
    epoch == 1 → rate = 0.4
    epoch >= 2 → rate = 0.3
    """
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            new_rate = 0.5
        elif epoch == 1:
            new_rate = 0.4
        else:
            new_rate = 0.3

        # Пробегаем по всем слоям модели, находим Dropout-слои и меняем rate
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.rate = new_rate
        print(f"\n>> DynamicDropoutCallback: Set Dropout rate = {new_rate} at epoch {epoch}")


def load_pairs(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    y = np.array([d["label"] for d in data], dtype="float32")
    return s1, s2, y


def build_siamese_lstm(vocab_size: int, embedding_dim: int, maxlen: int) -> Model:
    """
    Создаёт Siamese-LSTM модель без L2.
    Dropout инициализируется, но будет меняться динамически через колбэк.
    """
    in1 = Input(shape=(maxlen,), name="input_1")
    in2 = Input(shape=(maxlen,), name="input_2")

    embed = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="emb"
    )
    lstm = LSTM(128, name="lstm")

    v1 = lstm(embed(in1))
    v2 = lstm(embed(in2))

    diff = Lambda(lambda x: abs(x[0] - x[1]), name="abs_diff")([v1, v2])
    mult = Multiply(name="mult")([v1, v2])
    merged = Concatenate(name="features")([diff, mult])

    # Начальный rate=0.5, но потом колбэк его скорректирует
    x = Dropout(0.5)(merged)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[in1, in2], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main(
        NUM_WORDS=50000,
        MAXLEN=40,
        EMBEDDING_DIM=100,
        BATCH_SIZE=256,
        EPOCHS=10,
        VALID_SPLIT=0.1
    ):
    BASE_DIR = Path(__file__).resolve().parent.parent
    pairs_path = BASE_DIR / "data" / "train_pairs.json"
    tokenizer_path = BASE_DIR / "models" / "tokenizer_dynamic_dropout.pkl"
    model_path = BASE_DIR / "models" / "lstm_model_dynamic_dropout.h5"

    print(f"Loading pairs from {pairs_path} …")
    s1, s2, y = load_pairs(pairs_path)
    print(f"Loaded {len(y)} pairs.")

    # Строим Tokenizer
    print("Building tokenizer …")
    tokenizer = build_tokenizer(s1 + s2, num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Токенизация и паддинг
    print("Tokenizing and padding sequences …")
    X1 = texts_to_padded_sequences(tokenizer, s1, MAXLEN)
    X2 = texts_to_padded_sequences(tokenizer, s2, MAXLEN)

    # train/val split
    idx = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(idx)
    split_at = int((1 - VALID_SPLIT) * len(idx))
    train_idx, val_idx = idx[:split_at], idx[split_at:]

    X1_train, X2_train, y_train = X1[train_idx], X2[train_idx], y[train_idx]
    X1_val, X2_val, y_val = X1[val_idx], X2[val_idx], y[val_idx]

    print("Building Siamese-LSTM model (dynamic dropout) …")
    model = build_siamese_lstm(
        vocab_size=NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        maxlen=MAXLEN
    )
    model.summary()

    dynamic_dropout_cb = DynamicDropoutCallback()

    print("Starting training …")
    ckpt = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[ckpt, early, dynamic_dropout_cb]
    )

    history_dict = history.history
    with open(BASE_DIR / "models" / "history_dynamic_dropout.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=2)

    print("Training history saved to models/history_dynamic_dropout.json")
    print(f"Training finished. Best model saved to {model_path}")


if __name__ == "__main__":
    main()
