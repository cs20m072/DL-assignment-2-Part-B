import argparse
import torch

import wandb

image_width = 256
image_height = 256
input_channels = 3
output_classes = 10
training_directory = '/data/inaturalist_12K/train'
test_directory = '/data/nature_12K/inaturalist_12K/val'
labels = ["Amphibia", "Arachnida", "Fungi", "Mammalia", "Plantae", "Animalia", "Aves", "Insecta", "Mollusca",
          "Reptilia"]

device = ("cuda" if torch.cuda.is_available() else "cpu")

relu = "relu"
sigmoid = "sigmoid"
tanh = "tanh"

# optimizers
adam = 'adam'
nadam = 'nadam'
sgd = 'sgd'
momentum = 'momentum'
nesterov = 'nesterov'
rms = 'rmsprop'

no_conv_filters = [32,32,32,32,32]
conv_filters = [3, 3, 3, 3, 3]  # size of filter 3*3
activation_list = [relu, relu, relu, relu, relu]
conv_strides = [1, 1, 1, 1, 1]
conv_padding = [1, 1, 1, 1, 1]
maxpool_filters = [2, 2, 2, 2, 2]
maxpool_strides = [2, 2, 2, 2, 2]
maxpool_padding = [1, 1, 1, 1, 1]
batch_norm = "yes"

fcn_dropout = 0.3
cnn_dropout = 0.1

# linear layer configuration
linear = 1000
activation = relu

# training configuration
learning_rate = 0.01
batch_size = 32
gradient_accumalation = 2
shuffle = True
epochs = 3
optimizer = adam
loss = 'cross-entropy'
momentum = 0.9
weight_decay = 0
augment_data = True

# test configuration:
test_batch_size = 8

# data loader configuration
num_workers = 4

# pretrained model
model_name = "resnet"
freeze_cnn=False

parser = argparse.ArgumentParser("dl assignment 2")
parser.add_argument("--epochs", required = False , type=int, default= epochs)
parser.add_argument("--learning_rate", required = False, type=float, default=learning_rate)

parser.add_argument("--optimizer", required = False, default=optimizer)
parser.add_argument("--weight_decay", required = False,type=float, default=weight_decay)

parser.add_argument("--model_name", required = False, default= model_name)
parser.add_argument("--freeze_cnn", required = False, default= freeze_cnn , type=bool)

args = vars(parser.parse_args())



learning_rate=args["learning_rate"]
optimizer=args["optimizer"]
weight_decay=args["weight_decay"]
epochs =args["epochs"]

model_name = args["model_name"]
freeze_cnn = args["freeze_cnn"]