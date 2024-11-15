import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

os.chdir('E:/X/data/polymorph')
# Set the path for the input directory
input_dir = 'E:/X/polymorph/data-5' 
replacable_file_path = 'E:/X/polymorph/sim-x'

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

# Reshape the data to have 4 dimensions
data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))

# Get the shape of the input data
num_samples = data.shape[1]
num_channels = data.shape[2]
num_classes = len(lead_names)

model = tf.keras.Sequential([ layers.Conv2D(32, kernel_size=(3, 1), 
                            activation='relu', 
                            input_shape=(num_samples, num_channels, 1)),
                            layers.MaxPooling2D(pool_size=(2, 1)),
                            layers.Conv2D(64, kernel_size=(3, 1),activation='relu'),
                            layers.MaxPooling2D(pool_size=(2, 1)),
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
epochs = 10

model.fit(data, labels, batch_size=batch_size, epochs=epochs)

# Load the replacable ECG data
replacable_data = np.load(replacable_file_path)

# Generate simulated ECG data based on the replacable data
num_leads = len(lead_names)
num_samples = replacable_data[lead_names[0]].shape[0]
simulated_data = np.zeros((num_samples, num_leads, num_channels, 1))
prediction = model.predict(simulated_data.reshape(num_samples, num_leads, num_channels, 1))
lead_values = np.argmax(prediction, axis=-1)
simulated_data[:, :, 0, 0] = lead_values

print("Simulation completed.")

# Overwrite the replacable ECG data with the simulated data
for lead_name in lead_names:
    replacable_data[lead_name] = simulated_data[:, lead_names.index(lead_name), 0, 0]

# Save the modified replacable ECG data
np.savez(replacable_file_path, **replacable_data)

# Reshape the simulated data
simulated_data = np.transpose(simulated_data, (1, 0, 2, 3))

# Plot the simulated ECG 12-lead data
fig, axs = plt.subplots(nrows=num_leads, ncols=1, figsize=(200, 100))

for i, ax in enumerate(axs):
    ax.plot(simulated_data[:, i, 0, 0]) 
    ax.set_title(f'{lead_names[i]} - Lead {i + 1}')

plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.suptitle('Simulated ECG 12-Lead Data', fontsize=36)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()