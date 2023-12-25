import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Loading the CSV files
train_df = pd.read_csv('train.csv')

img_size = (224, 224)

# Function to load images
def load_images(df, path):
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = f"{path}/{row['id']}.jpg"
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(row['species'])

    return np.array(images), np.array(labels)

# Converting species names to numerical labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df['species'])
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=len(label_encoder.classes_))

# Loading training images
train_images, _ = load_images(train_df, 'images')

# Building the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Adjust output layer units


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=32)
model.save('IMG_leaf_classifier_model.h5')
#model.save('IMG_leaf_classifier_model.keras')




# Evaluate the model on the test set
# test_images, _ = load_images(test_df, 'path/to/test/images')
# test_predictions = model.predict(test_images)
# Add code to handle test predictions as needed
