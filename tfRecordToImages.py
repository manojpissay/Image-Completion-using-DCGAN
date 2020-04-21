import tensorflow as tf
import io
import PIL
import matplotlib.pyplot as plt
import imageio
import os

record_iterator = tf.python_io.tf_record_iterator("/content/drive/My Drive/Image Completion using DCGAN/data/pokemon/pokemon.tfrecords")
i=0
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    i+=1
    image = example.features.feature["image_raw"].bytes_list.value[0]
    height= example.features.feature["height"]
    width= example.features.feature["width"]
    image=tf.cast(tf.decode_raw(image,tf.uint8),tf.float32)
    image = tf.reshape(image, [100, 100, 3])
    image=tf.cast(image,tf.uint8)
    jpeg_bin_tensor = tf.image.encode_jpeg(image)
    if not os.path.exists('original_images'):
        os.makedirs('original_images')
    p=os.path.join("original_images/{:02d}.jpg".format(i))
    with tf.Session() as sess:
        jpeg_bin = sess.run(jpeg_bin_tensor)
        jpeg_str = io.BytesIO(jpeg_bin)
        jpeg_image = PIL.Image.open(jpeg_str)
        jpeg_image=jpeg_image.save(p)