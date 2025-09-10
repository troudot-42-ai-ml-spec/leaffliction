from .hyperparams import EPOCHS


def train_model(model, train_set, val_set, test_set):
    """
        Compile and train CNN, then evaluate it against test set.
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    print("⏳ Training model...")
    history = model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set
    )
    # --- Model Evaluation ---
    print("⏳ Evaluating model...")
    loss, accuracy = model.evaluate(test_set)
    print(f"➡️  Test Accuracy: ✨ {accuracy * 100:.2f}% ✨")

    return history
