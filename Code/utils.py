import PIL.Image
import tensorflow as tf
import numpy as np
from loss import style_loss,content_loss,total_variation_loss
import time


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32) # Casts a tensor to a new type.
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # Adding a new axis because input must be 4-dimensional[405,512,3] [Op:Conv2D]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0) # Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed

    plt.imshow(image)
    if title:
        plt.title(title)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def train(model,content_image,style_image,style_weight,content_weight,total_variation_weight,opt,epochs,steps_per_epoch):

    # Assign current_image to the content image
    current_image = tf.Variable(content_image)


    # Targets
    style_targets = model(style_image)['style']
    content_targets = model(content_image)['content']

    # Gradient Descent
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            with tf.GradientTape() as tape:
                outputs = model(current_image)
                style_outputs = outputs['style']
                content_outputs = outputs['content']
                loss = style_loss(style_outputs,style_targets,style_weight)
                loss += content_loss(content_outputs,content_targets,content_weight)
                loss += total_variation_loss(current_image, total_variation_weight)

            grad = tape.gradient(loss, current_image)
            opt.apply_gradients([(grad, current_image)])
            current_image.assign(clip_0_1(current_image))
            print(".", end='')

        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return current_image
