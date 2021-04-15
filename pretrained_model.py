
import torchvision
from torch.nn import Linear, ReLU, Sequential, Conv2d, Dropout, BatchNorm1d

import configuration


def getLinearLayer(classifier_input):
    i=0
    linear_layers = Sequential()
    linear_layers.add_module("linear_{}".format(i), Linear(classifier_input, configuration.linear))
    linear_layers.add_module("dropout_{}".format(i), Dropout(configuration.fcn_dropout))
    linear_layers.add_module("activation_{}".format(i), ReLU(inplace=True))
    if configuration.batch_norm:
        linear_layers.add_module("batch_norm{}".format(i), BatchNorm1d(configuration.linear))
    linear_layers.add_module("output_{}".format(i), Linear(configuration.linear, configuration.output_classes))
    return linear_layers


"""
1) remove FC layer and add new FC layer update only FC layer
2) append extra FC layer update only FC layer
3) remove FC layer and add new FC layer update only FC layer and last 3 conv layers
4) append extra FC layer update only FC layer and last 3 conv layers
"""
def set_grad_false(model_ft, index=-1):
    l = list(model_ft.parameters())
    if configuration.freeze_cnn:
        for param in l[:index]:
            param.requires_grad = False
    else:
        for param in l:
            param.requires_grad = False



def getModel():
    index=-1
    if configuration.model_name== "inception":
        model=torchvision.models.inception_v3(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier_input = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = getLinearLayer(classifier_input)

        classifier_input = model.fc.in_features
        model.fc =  getLinearLayer(classifier_input)

        configuration.image_height=299
        configuration.image_width=299
        return model

    if configuration.model_name == "resnet":
        model_ft = torchvision.models.resnet18(pretrained=True)
        if configuration.freeze_cnn:
            index=48

        set_grad_false(model_ft,index)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = getLinearLayer(num_ftrs)
        configuration.image_height = 224
        configuration.image_width = 224
        return model_ft

    if configuration.model_name == "vgg":
        model_ft = torchvision.models.vgg11_bn(pretrained=True)
        if configuration.freeze_cnn:
            index=32
        set_grad_false(model_ft, index)
        model_ft.classifier.add_module("linear_last", Linear(1000, configuration.output_classes))
        configuration.image_height = 224
        configuration.image_width = 224
        return model_ft

    if configuration.model_name == "densenet":
        model_ft = torchvision.models.densenet201(pretrained=True)
        set_grad_false(model_ft)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier= Linear(num_ftrs, configuration.output_classes)
        configuration.image_height = 224
        configuration.image_width = 224
        return model_ft

    if configuration.model_name == "alexnet":
        model_ft = torchvision.models.alexnet(pretrained=True)
        if configuration.freeze_cnn:
            index = 10
        set_grad_false(model_ft, index)

        model_ft.classifier.add_module("linear_last",Linear(1000, configuration.output_classes))
        configuration.image_height = 224
        configuration.image_width = 224
        return model_ft

    if configuration.model_name == "squeeznet":
        model_ft = torchvision.models.squeezenet1_0(pretrained=True)
        if configuration.freeze_cnn:
            index = 48
        set_grad_false(model_ft, index)
        model_ft.classifier[1] =  Conv2d(512, configuration.output_classes, kernel_size=(1,1), stride=(1,1))
        configuration.image_height = 224
        configuration.image_width = 224
        return model_ft

