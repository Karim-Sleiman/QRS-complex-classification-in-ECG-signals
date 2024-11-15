import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Set the path for the input directory 
input_dir = 'E:/X/preprocessed-data'

# Step 1: Load the preprocessed data
data = []
labels = []
lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
for filename in os.listdir(input_dir): 
    if filename.endswith('.npz'):
        npz_data = np.load(os.path.join(input_dir, filename))
        for lead_name in lead_names:
            lead_data = npz_data[lead_name]
            num_samples = lead_data.shape[0]
            data.append(lead_data)
            labels.append(lead_names.index(lead_name))

# Convert the data and labels to arrays
data = np.array(data)
labels = np.array(labels)

# Reshape the data to have 3 dimensions
data = np.reshape(data, (data.shape[0], data.shape[1], 1))

# Get the shape of the input data
num_samples = data.shape[1]
num_channels = data.shape[2]
num_classes = len(lead_names)

model = tf.keras.Sequential([ layers.Conv1D(32, kernel_size=3, activation='relu', 
                            input_shape=(num_samples, num_channels)),
                            layers.MaxPooling1D(pool_size=2), 
                            layers.Conv1D(64, kernel_size=3, activation='relu'),
                            layers.MaxPooling1D(pool_size=2),
                            layers.Flatten(),
                            layers.Dense(128, activation='relu'), 
                            layers.Dense(num_classes, activation='softmax')
])

# Compile the model 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 20
model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(data, labels) 
print("Test Loss:", loss) 
print("Test Accuracy:", accuracy)