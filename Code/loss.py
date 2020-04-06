import tensorflow as tf
import numpy as np

"""
Total loss = content_loss + style_loss + Total_variation_loss
"""



def style_loss(style_outputs,style_targets,style_weight):


    num_style_layers = len(style_outputs.keys())
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    return style_loss

def content_loss(content_outputs,content_targets,content_weight):

    num_content_layers = len(content_outputs.keys())
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight/num_content_layers
    return content_loss


def total_variation_loss(image,total_variation_weight):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    loss = tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))
    loss *= total_variation_weight
    return loss
