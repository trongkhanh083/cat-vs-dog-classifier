from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.model import convolutional_model
from src.dataset import prepare_data, get_generator
import os
import pickle

### Model Training
# Training custom CNN
def train():
    
    # split & prepare data
    prepare_data()

    # get generator
    train_generator, val_generator, _ = get_generator()
    
    # build model
    model = convolutional_model()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'checkpoints/best_model_scratch.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator, 
        epochs=30, 
        validation_data=val_generator,
        callbacks=callbacks
    )

    os.makedirs('outputs', exist_ok=True)
    history_path = f"outputs/history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Save training history to {history_path}")
    return history

def main():
    train()

if __name__=="__main__":
    main()

# Training VGG16
# callbacks = [
#     EarlyStopping(
#         monitor='val_accuracy',
#         patience=5, 
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ModelCheckpoint(
#         'checkpoints/best_model.h5', 
#         monitor='val_accuracy',
#         save_best_only=True,
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5, 
#         patience=2,
#         min_lr=1e-6,
#         verbose=1
#     )
# ]

# history = model.fit(
#     train_generator,
#     epochs=30,
#     validation_data=val_generator,
#     callbacks=callbacks
# )