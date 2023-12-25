import pandas as pd
import numpy as np
from keras import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Convolution1D
from tensorflow.python.keras.layers import Activation, Concatenate, Input
from keras.optimizers import Adam


# Function to load and preprocess images
def load_images(df, path):
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = f"{path}/{row['id']}.jpg"
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(row['species'])

    return np.array(images), np.array(labels)


def training():
    df_train = pd.read_csv('train.csv')

    ######################################
    # Image CNN

    model_2D = Sequential()

    # Convert species names to numerical labels using LabelEncoder
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(df_train['species'])
    y_train = to_categorical(train_labels_encoded, num_classes=len(label_encoder.classes_))

    # Load and preprocess training images
    train_images, _ = load_images(df_train, 'images')

    model_2D.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model_2D.add(MaxPooling2D(pool_size=(2, 2)))

    model_2D.add(Conv2D(64, (3, 3), activation='relu'))
    model_2D.add(MaxPooling2D(pool_size=(2, 2)))

    model_2D.add(Conv2D(128, (3, 3), activation='relu'))
    model_2D.add(MaxPooling2D(pool_size=(2, 2)))

    model_2D.add(Flatten())

    model_2D.add(Dense(512, activation='relu'))
    model_2D.add(Dropout(0.5))
    model_2D.add(Dense(len(label_encoder.classes_), activation='softmax'))

    ######################################
    # 1D CNN

    train = df_train.drop(['species', 'id'], axis=1)

    scaler = StandardScaler().fit(train.values)
    scaled_train = scaler.transform(train.values)

    nb_features = 64
    nb_classes = 99

    X_train_r = np.zeros((len(scaled_train), nb_features, 3))
    X_train_r[:, :, 0] = scaled_train[:, :nb_features]  ##margin type features take first 64
    X_train_r[:, :, 1] = scaled_train[:, nb_features:128]  ##shape type features take second 64 (64*2)
    X_train_r[:, :, 2] = scaled_train[:, 128:]  ##texture type features take last 64 from number (64*2) to end

    model_1D = Sequential()
    model_1D.add(Convolution1D(512, 1, input_shape=(nb_features, 3)))
    model_1D.add(Activation('relu'))
    model_1D.add(Flatten())
    model_1D.add(Dropout(0.4))
    model_1D.add(Dense(2048, activation='relu'))
    model_1D.add(Dense(1024, activation='relu'))
    model_1D.add(Activation('softmax'))
    model_1D.add(Dense(nb_classes))

    ######################################
    # Merged CNN

    flatten_2D = Flatten()(model_2D.layers[-1].output)
    output_2D = Dense(512, activation='relu')(flatten_2D)


    flatten_1D = model_1D.layers[-3].output
    output_1D = Dense(len(label_encoder.classes_), activation='softmax')(flatten_1D)

    concatenated = Concatenate()([output_2D, output_1D])

    merged_model = Model(inputs=[model_2D.input, model_1D.input], outputs=concatenated)
    merged_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    merged_model.fit([train_images, X_train_r], y_train, epochs=10, batch_size=32, validation_split=0.2)

    '''
    input_2D = Input(shape=(224, 224, 3))
    input_1D = Input(shape=(nb_features, 3))

    # Concatenate the flattened output of model_2D with the output of model_1D
    merged = Concatenate()([model_2D, model_1D])

    output_layer = Dense(nb_classes, activation='softmax')(merged)
    merged_model = Model(inputs=[input_2D, input_1D], outputs=output_layer)

    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    merged_model.fit(train_images, y_train, epochs=10, batch_size=32)
    merged_model.save('merged_leaf_classifier_model.keras')

    input_2D = Input(shape=(224, 224, 3))
    output_2D = model_2D(input_2D)

    input_1D = Input(shape=(nb_features, 3))
    output_1D = model_1D(input_1D)

    merged = Concatenate()([output_2D, output_1D])
    output_layer = Dense(nb_classes, activation='softmax')(merged)
    merged_model = Model(inputs=[input_2D, input_1D], outputs=output_layer)

    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    merged_model.fit(train_images, y_train, epochs=10, batch_size=32)
    merged_model.save('merged_leaf_classifier_model.keras')'''

training()
