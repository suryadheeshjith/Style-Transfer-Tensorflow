import tensorflow as tf
from utils import gram_matrix
import numpy as np

"""
Total loss = content_loss + style_loss + Total_variation_loss
"""


def content_loss(content_weight, content_current, content_original):

    loss = content_weight*tf.reduce_sum((content_current-content_original)**2)
    return loss



def style_loss(feats, style_layers, style_targets, style_weights):

    i =0
    loss = 0.0
    for layer in style_layers:
        g_matx = gram_matrix(feats[layer])
        loss+= style_weights[i]*tf.math.reduce_sum((g_matx-style_targets[i])**2)
        i+=1
    return loss

def tv_loss(img, tv_weight):

    sum1 = (img[:, 1:, :, :]-img[:, :-1, :, :])
    sum2 = (img[:, :, 1:, :]-img[:, :, :-1, :])
    loss = tv_weight*(np.sum(np.power(sum1, 2))+np.sum(np.power(sum2, 2)))
    return loss
