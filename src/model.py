import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, ReLU, Flatten, BatchNormalization
from tensorflow.keras import Input, Model

### Model Building
# Option 1: Build model from scratch

def convolutional_model(input_shape=(128,128,3)):
    inputs = Input(input_shape)

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    # Classification head
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


### Option 2: Transfer learning

# Transfer learning from VGG16
# from tensorflow.keras.applications import VGG16

# base_model = VGG16(weights='imagenet', 
#                   include_top=False, 
#                   input_shape=(128, 128, 3))

# # Freeze convolutional layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom head
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.5)(x)

# outputs = Dense(1, activation='sigmoid')(x)

# model = Model(base_model.input, outputs)

# model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

# model.summary()