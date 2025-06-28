from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from src.dataset import get_generator
import pickle

### Model Evaluation
def evaluate():
    # Test for custom CNN
    model = load_model('checkpoints/best_model_scratch.h5')
    _, _, test_generator = get_generator()
    
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Another metrics test
    y_prob = model.predict(test_generator)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = test_generator.classes
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    auc = roc_auc_score(y_true, y_prob)
    print("Test AUC:", auc)


    # Plot accuracy and loss curve
    history_path = "output/history.pkl"
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label='Train acc')
    plt.plot(epochs, val_acc, label='Val acc')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/accuracy.png')
    plt.close()
    print(f"Saved accuracy curves to {'output/accuracy.png'}")

    plt.figure()
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/loss.png')
    plt.close()
    print(f"Saved loss curves to {'output/loss.png'}")
    

def main():
    evaluate()

if __name__=="__main__":
    main()
