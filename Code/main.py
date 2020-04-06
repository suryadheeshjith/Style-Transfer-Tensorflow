import tensorflow as tf
import argparse
from utils import load_img,train,tensor_to_image
from Style_class import StyleContentModel



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp',"--content_path",dest="content_path", default='../Data/Content/waves.jpg', help="Content file path. Default='../Data/Content/waves.jpg'")
    parser.add_argument('-sp',"--style_path",dest="style_path", default='../Data/Styles/wave.jpg', help="Style file path. Default='../Data/Styles/wave.jpg'")
    parser.add_argument('-op',"--output_path",dest="output_path", default='../Data/Output/1.jpg', help="Output file path. Default='../Data/Content/1.jpg'")
    parser.add_argument('-cpl',"--content_path_layer",dest="content_path_layer", default='../Data/Layers/content.txt', help="Content text file with layer name. Default='../Data/Layers/content.txt'")
    parser.add_argument('-spl',"--style_path_layers",dest="style_path_layers", default='../Data/Layers/style.txt', help="Style text file with layer names. Default='../Data/Layers/style.txt'")
    parser.add_argument('-sw',"--style_weight", dest="style_weight", type=float, default=1e-2, help="Style weights. Default=1e-2")
    parser.add_argument('-cw',"--content_weight", dest="content_weight", type=float, default=1e4, help="Content weights. Default=1e4")
    parser.add_argument('-tw',"--total_variation_weight", dest="total_variation_weight", type=int, default=30, help="Total Variation weight. Default=30")
    parser.add_argument('-lr',"--learning_rate", dest="learning_rate", type=float, default=0.02, help="Learning Rate in Adam. Default=0.02")
    parser.add_argument('-b',"--beta1", dest="beta_1", type=float, default=0.99, help="Beta in Adam. Default=0.99")
    parser.add_argument('-e',"--epochs", dest="epochs", type=int, default=10, help="Epochs. Default=10")
    parser.add_argument('-spe',"--steps_per_epoch", dest="steps_per_epoch", type=int, default=100, help="Steps per epoch. Default=100")


    args = parser.parse_args()
    return args


if __name__ == "__main__":


    #Defaults
    content_layers_default = ['block5_conv2']
    style_layers_default = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

    # Argument Parsing
    args = parse_args()
    print("\n\nArguments Used : "+str(args)+"\n\n\n")
    content_path = args.content_path
    style_path =  args.style_path
    output_path = args.output_path
    c_layers_path = args.content_path_layer
    s_layers_path = args.style_path_layers
    style_weight = args.style_weight
    content_weight = args.content_weight
    total_variation_weight = args.total_variation_weight
    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch



    # Extract Images
    content_image = load_img(content_path)
    style_image = load_img(style_path)


    # Extract layer names
    content_layers = []
    style_layers = []
    try:
        f1 = open(c_layers_path, "r")
        for layer in f1:
            layer = layer.rstrip()
            content_layers.append(layer)

    except:
        content_layers = content_layers_default

    try:
        f2 = open(s_layers_path, "r")
        for layer in f2:
            layer = layer.rstrip()
            style_layers.append(layer)
    except:
        style_layers = style_layers_default

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate, beta_1, epsilon=0.1)

    # Model
    model = StyleContentModel(style_layers, content_layers)

    # Train
    output = train(model,content_image,style_image,style_weight,content_weight,total_variation_weight,opt,epochs,steps_per_epoch)

    tensor_to_image(output).save(output_path)
