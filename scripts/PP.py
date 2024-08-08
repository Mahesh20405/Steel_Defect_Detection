import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CSV file
df = pd.read_csv("E:/PJ/AI VT12/D/train.csv")

# Convert ClassId column to string
df['ClassId'] = df['ClassId'].astype(str)

# Define ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="E:/PJ/AI VT12/D/TR",
    x_col="ImageId",
    y_col="ClassId",
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical",
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="E:/PJ/AI VT12/D/TR",
    x_col="ImageId",
    y_col="ClassId",
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical",
    subset='validation'
)
