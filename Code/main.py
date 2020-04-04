import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib
#matplotlib.use('TkAgg') # Require matplotlib output
matplotlib.use('PS') # No matplotlib output
import matplotlib.pyplot as plt
import tensorflow as tf


from utils import load_image, preprocess_image, deprocess_image, extract_features, gram_matrix
from loss import content_loss,style_loss,tv_loss
from SqueezeNet import SqueezeNet



def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, model, path, init_random = False):

    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = extract_features(content_img[None], model)
    content_target = feats[content_layer]

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    s_feats = extract_features(style_img[None], model)
    style_targets = []
    # Compute list of TensorFlow Gram matrices
    for idx in style_layers:
        style_targets.append(gram_matrix(s_feats[idx]))

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180
    max_iter = 400

    step = tf.Variable(0, trainable=False)
    boundaries = [decay_lr_at]
    values = [initial_lr, decayed_lr]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize the generated image and optimization variables

    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess_image(content_img))
    axarr[1].imshow(deprocess_image(style_img))
    plt.show()
    plt.figure()

    # Initialize generated image to content image
    if init_random:
        initializer = tf.random_uniform_initializer(0, 1)
        img = initializer(shape=content_img[None].shape)
        img_var = tf.Variable(img)
        print("Intializing randomly.")
    else:
        img_var = tf.Variable(content_img[None])
        print("Initializing with content image.")

    for t in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(img_var)
            feats = extract_features(img_var, model)
            # Compute loss
            c_loss = content_loss(content_weight, feats[content_layer], content_target)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(img_var, tv_weight)
            loss = c_loss + s_loss + t_loss
        # Compute gradient
        grad = tape.gradient(loss, img_var)
        optimizer.apply_gradients([(grad, img_var)])

        img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
            plt.axis('off')
            plt.show()
    print('Iteration {}'.format(t))
    plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
    plt.savefig(path)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    squeezenet=SqueezeNet()
    squeezenet.load_weights('../weights/squeezenet.ckpt')
    squeezenet.trainable=False

    path_to_out = '../Data/Output/saved9.png'


    params1 = {
    'content_image' : '../Data/Content/mit.jpg',
    #'style_image' : '../Data/Styles/composition_vii.jpg',
    #'style_image' : '../Data/Styles/the_scream.jpg',
    #'style_image' : '../Data/Styles/muse.jpg',
    #'style_image' : '../Data/Styles/udnie.jpg',
    'style_image' : '../Data/Styles/wave.jpg',
    'image_size' : 192,
    'style_size' : 224,
    'content_layer' : 2,
    'content_weight' : 3e-2,
    'style_layers' : (0, 3, 5, 6),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2,
    'model' : squeezenet,
    'path' : path_to_out
}

    style_transfer(**params1)
