from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, \
    Dropout, Sigmoid, Tanh, BatchNorm1d, Flatten

import configuration



def get_activation_from_name(a):
    act = ReLU(inplace=True)
    if a == configuration.sigmoid:
        act = Sigmoid()
    if a == configuration.tanh:
        act = Tanh()
    return act

class Net(Module):

    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential()
        input_channels = configuration.input_channels
        width = configuration.image_width
        height = configuration.image_height


        i = 1
        for k, f, a, s, p, mf, ms, mp in zip(
                configuration.no_conv_filters, configuration.conv_filters, configuration.activation_list,
                configuration.conv_strides, configuration.conv_padding, configuration.maxpool_filters,
                configuration.maxpool_strides, configuration.maxpool_padding):

            self.cnn_layers.add_module("conv_2d_{}".format(i),
                                       Conv2d(input_channels, k, kernel_size=f, stride=s, padding=p))
            self.cnn_layers.add_module("dropout{}".format(i), Dropout(configuration.cnn_dropout))
            width = (width - f + 2 * p) / s + 1
            height = (height - f + 2 * p) / s + 1
            input_channels = k

            self.cnn_layers.add_module("activation_{}".format(i), get_activation_from_name(a))

            if configuration.batch_norm:
                self.cnn_layers.add_module("batch_norm{}".format(i), BatchNorm2d(input_channels))

            self.cnn_layers.add_module("max_pool_{}".format(i), MaxPool2d(kernel_size=mf, stride=ms, padding=mp))

            width = int((width - mf + 2 * mp) / ms + 1)
            height =int((height - mf + 2 * mp) / ms + 1)

            i += 1

        self.linear_layers = Sequential()
        self.linear_layers.add_module("flatten",Flatten())
        self.linear_layers.add_module("linear_{}".format(i),Linear(input_channels * width * height, configuration.linear))
        self.linear_layers.add_module("dropout_{}".format(i),Dropout(configuration.fcn_dropout))
        self.linear_layers.add_module("activation_{}".format(i),get_activation_from_name(configuration.relu))

        if configuration.batch_norm:
            self.linear_layers.add_module("batch_norm{}".format(i),BatchNorm1d(configuration.linear))

        self.linear_layers.add_module("output_{}".format(i),Linear(configuration.linear, configuration.output_classes))
        #self.linear_layers.add_module("softmax_{}".format(i),Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x
