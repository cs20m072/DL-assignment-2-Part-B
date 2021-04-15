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

# CNN network configuration for each layer
values = [[32, 32, 32, 32, 32], [64, 64, 64, 64, 64], [32, 64, 64, 64, 128], [64, 64, 64, 128, 128],
          [32, 32, 32, 64, 64], [32, 32, 32, 64, 128],
          [64, 64, 32, 32, 32], [64, 32, 32, 32, 32], [64, 64, 64, 32, 32]]

no_conv_filters = values[0]
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
epochs = 10
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



hyperparameter_defaults = dict(
    model_name="inception",
    freeze_cnn=True,
    epochs=20,
    learning_rate=0.0001,
    optimizer="sgd",
    weight_decay=0.001
)

wandb.init(config=hyperparameter_defaults, project="cnn-assignment-2-transfer-learning")
config = wandb.config


model_name=config.model_name
freeze_cnn=config.freeze_cnn


epochs=config.epochs
learning_rate=config.learning_rate
optimizer=config.optimizer
weight_decay=config.weight_decay


