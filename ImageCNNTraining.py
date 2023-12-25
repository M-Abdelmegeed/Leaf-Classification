import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from ImageCNN import label_encoder

# Load the CSV files
test_df = pd.read_csv('test.csv')

# Function to load test images
def load_test_images(df, path):
    images = []

    for idx, row in df.iterrows():
        img_name = row['id'].astype(np.int64)
        img_path = f"{path}/{img_name}.jpg"
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)

    return np.array(images)


test_images = load_test_images(test_df, 'images')

model = load_model('leaf_classifier_model.h5')
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
predicted_species = label_encoder.inverse_transform(predicted_classes)
result_df = pd.DataFrame({'id': test_df['id'], 'predicted_species': predicted_species})

# Saving the results to a CSV file
result_df.to_csv('test_predictions.csv', index=False)




