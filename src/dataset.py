import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Split data to train/val/test
def prepare_data(raw_dir='data/raw', output_dir='data', random_seed=42):

    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['cat', 'dog']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
            print(f"Create directory: {output_dir}/{split}/{class_name}")


    # Split train/val/test 80/10/10
    def split_class(class_name):
        class_path = os.path.join(raw_dir, class_name)

        # Get all image files
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nFound {len(images)} {class_name} images")

        # Split into 80/10/10
        train, test = train_test_split(images, test_size=0.2, random_state=random_seed)
        val, test = train_test_split(test, test_size=0.5, random_state=random_seed)

        # Helper function to copy files
        def copy_files(files, split_name):
            for f in files:
                src = os.path.join(class_path, f)
                dst = os.path.join(output_dir, split_name, class_name, f)
                shutil.copy2(src, dst)

        copy_files(train, 'train')
        copy_files(val, 'val')
        copy_files(test, 'test')

        print(f"Split {class_name} images:")
        print(f" Training {len(train)}")
        print(f" Validation {len(val)}")
        print(f" Test {len(test)}")    


    # Process both class
    split_class('cat')
    split_class('dog')


    # Verification
    print(f"\nFinal counts:")
    for split in ['train', 'val', 'test']:
        for class_name in ['cat', 'dog']:
            count = len(os.listdir(os.path.join(output_dir, split, class_name)))
            print(f"{split}/{class_name}: {count} images")



## Data Preprocessing
def get_generator(output_dir='data', target_size=(128,128), batch_size=32):
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=[0.8, 1.2],
        brightness_range=(0.8, 1.2),
        channel_shift_range=15,
        horizontal_flip=True,
        fill_mode='reflect'
    )

    # Only rescale for val/test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(output_dir, 'train'),
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary'
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(output_dir, 'val'),
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary'
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(output_dir, 'test'),
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False # Important for evaluation
    )
    return train_generator, val_generator, test_generator
