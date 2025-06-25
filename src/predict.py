import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

### Predict and display image
def predict_and_display_test_set(test_dir, model, pred_path, num_images):

    # Get all test image paths
    cat_dir = os.path.join(test_dir, 'cat')
    dog_dir = os.path.join(test_dir, 'dog')
    
    cat_images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dog_images = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    all_images = cat_images + dog_images
    if num_images is not None:
        all_images = all_images[:num_images]
    
    # Create figure
    cols = 5  # Number of columns in display
    rows = int(np.ceil(len(all_images) / cols))
    plt.figure(figsize=(20, 5*rows))
    
    for i, img_path in enumerate(all_images):
        # Load and predict
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        
        pred = model.predict(x, verbose=0)[0][0]
        confidence = pred if pred > 0.5 else 1 - pred
        label = 'Dog' if pred > 0.5 else 'Cat'
        true_label = 'Dog' if 'dog' in img_path.lower() else 'Cat'
        
        # Plot
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {label} ({confidence*100:.1f}%)", 
                 color='green' if label == true_label else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(pred_path)
    plt.close()
    print(f"Save prediction to {pred_path}")

def main():
    # Load model checkpoint
    model = load_model('checkpoints/best_model_scratch.h5')

    # Predict on test set
    predict_and_display_test_set('data/test', model, 'outputs/prediction.png', num_images=100)

    # Optional for your image (create realworld_test folder contain cat and dog subfolder)
    predict_and_display_test_set('data/realworld_test', model, 'outputs/pred_realworld.png', num_images=50)


if __name__=="__main__":
    main()