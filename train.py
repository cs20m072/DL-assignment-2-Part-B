import warnings

warnings.filterwarnings("ignore")

from torch.optim import Adam, SGD
from dataLoader import *
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
import configuration
import wandb
import pretrained_model


def get_optimizer_fun(model, opt):
    parameters=[]
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            parameters.append(param)
            print("\t", name)

    optimizer = Adam(parameters, lr=configuration.learning_rate, weight_decay=configuration.weight_decay)
    if opt == configuration.sgd:
        optimizer = SGD(parameters, configuration.learning_rate, momentum=configuration.momentum,
                        weight_decay=configuration.weight_decay)
    return optimizer


def get_chunk(iterable, chunk_size):
    result = []
    for item in iterable:
        result.append(item)
        if len(result) == chunk_size:
            yield tuple(result)
            result = []
    if len(result) > 0:
        yield torch.floatTensor(result)


def train(epoch):
    model.train()
    avg_loss_train = 0
    average_train_acc = 0
    average_val_acc = 0
    gradient_accumalation = configuration.gradient_accumalation

    scalar = GradScaler()

    # loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
    batch_idx = 0
    for (images, labels) in tqdm(train_dataloader):
        images = images.to(configuration.device)
        labels = labels.to(configuration.device)

        with autocast():
            if (configuration.model_name == "inception"):
                output_train, aux_outputs = model(images)
                loss1 = criterion(output_train, labels)
                loss2 = criterion(aux_outputs, labels)
                loss_train = loss1 + 0.4 * loss2
            else:
                output_train = model(images)
                loss_train = criterion(output_train, labels)

        scalar.scale(loss_train / gradient_accumalation).backward()

        avg_loss_train += loss_train.item()
        prob = (output_train.cpu()).argmax(dim=1)
        true_label = labels.cpu()
        acc = torch.sum(prob == true_label) / configuration.batch_size
        average_train_acc += acc

        if (batch_idx + 1) % gradient_accumalation == 0:
            scalar.step(optimizer)
            scalar.update()
            model.zero_grad()

        batch_idx += 1

    avg_loss_val = 0
    for (images, labels) in tqdm(validation_dataloader):
        images = images.to(configuration.device)  # move data to cuda memory
        labels = labels.to(configuration.device)
        with torch.no_grad():
            if configuration.model_name == "inception":
                output_val, aux_output = model(images)
                loss1 = criterion(output_val, labels)
                loss2 = criterion(aux_output, labels)
                loss_val = loss1 + 0.4 * loss2
            else:
                output_val = model(images)
                loss_val = criterion(output_val, labels)

            avg_loss_val += loss_val.item()

            prob = (output_val.cpu()).argmax(dim=1)
            true_label = labels.cpu()
            acc = torch.sum(prob == true_label) / configuration.test_batch_size
            average_val_acc += acc

    if (epoch+1)%5 ==0 or epoch==1:
        # print("epoch: ", epoch + 1, '\t', 'loss_train :', avg_loss_train / len(train_dataloader))
        # print("epoch: ", epoch + 1, '\t', 'loss_val :', avg_loss_val / len(validation_dataloader))
        # print("epoch: ", epoch + 1, '\t', 'acc_train :', average_train_acc / len(train_dataloader))
        # print("epoch: ", epoch + 1, '\t', 'acc_val:', average_val_acc / len(validation_dataloader))
        metric = {"val_accuracy": average_val_acc / len(validation_dataloader),
                  "train_accuracy:": average_train_acc / len(train_dataloader),
                  "val_loss:": avg_loss_val / len(validation_dataloader),
                  "train_loss:": avg_loss_train / len(train_dataloader)}
        print(metric)
        wandb.log(metric)


if __name__ == '__main__':

    model = pretrained_model.getModel()

    optimizer = get_optimizer_fun(model, configuration.optimizer)
    criterion = CrossEntropyLoss()
    train_dataloader, validation_dataloader = get_training_data_loaders()

    if configuration.device == 'cuda':
        print("cuda is available")
        model = model.cuda()
        criterion = criterion.cuda()

    for e in range(configuration.epochs):
        train(e)
