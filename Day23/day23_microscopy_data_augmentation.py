# microscopy_data_augmentation.py

"""
Microscopy image data augmentation for biotechnology applications using Keras and synthetic transformations.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

# Load a sample microscopy image
img = load_img('cell_sample.jpg')  # Replace with your microscopy image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Create an ImageDataGenerator tailored for microscopy
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=10,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True
)

# Output directory
output_dir = 'augmented_cells'
os.makedirs(output_dir, exist_ok=True)

# Generate and save augmented images
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='cell_aug', save_format='png'):
    i += 1
    if i > 12:
        break
