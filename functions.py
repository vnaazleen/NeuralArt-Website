import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np

import PIL.Image as Image
import IPython.display as display

import os
import uuid


def model(content, style):
	content_image_path = 'static/' + content + '.jpg'
	style_image_path = 'static/' + style + '.jpg'


	# Load content and style images (see example in the attached colab).
	content_image = plt.imread(content_image_path)
	style_image = plt.imread(style_image_path)

	os.remove(content_image_path)
	os.remove(style_image_path)

	# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
	content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
	style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
	style_image = tf.image.resize(style_image, (256, 256))

	# Load image stylization module.
	hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

	# Stylize image.
	outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
	stylized_image = outputs[0]
	transformed = str(uuid.uuid4())
	transformed_img_path = "static/" + transformed + '.jpg'

	f = open("filenames.txt", "a")
	f.write(transformed_img_path + '\n')
	f.close()

	tensor_to_image(stylized_image).save(transformed_img_path)
	return transformed


def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
		return Image.fromarray(tensor)


def delete_files():
    print("\n\n\n")
    f = open("filenames.txt", "r")
    imgs_to_delete = f.readlines()
    if len(imgs_to_delete) <= 1:
      print("only one img")
      pass
    else:
      # deleteing the last stylized imgs
      for img in imgs_to_delete[:-1]:
        print("deleted " + img)
        os.remove(img.strip())
        f.close()
        f = open("filenames.txt", "w")
        f.write(imgs_to_delete[-1])
    f.close()
    print("\n\n\n")

