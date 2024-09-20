import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved model weights
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_dim=4, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(36, activation='linear')
])

# Load weights
model.load_weights("dqn_model_bug_find.h5")  # Change the file name to your saved model

# Plot the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Display the plot using matplotlib
img = Image.open('model_plot.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
