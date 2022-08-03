import copy
import datetime
import re
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm


def readClassLabels(filePath="CamVid/label_colors.txt"):
    classes2values = {}
    values2classes = {}
    class2idx = {}
    idx2class = {}

    dir = os.path.dirname(__file__)

    newFilePath = os.path.join(dir, filePath)

    lines = ""

    with(open(newFilePath, "r") as file):
        lines = file.read().split("\n")

    for i, line in enumerate(lines):
        splitted = re.sub("\t\t", "\t", line).split("\t")
        values, classLabel = splitted[0], splitted[1]
        r, g, b = values.split(" ")
        r, g, b = int(r), int(g), int(b)

        class2idx[classLabel] = i
        idx2class[i] = classLabel
        values2classes[(r, g, b)] = i
        classes2values[i] = (r, g, b)

    return values2classes, classes2values, class2idx, idx2class


def showSample(sample):
    image, label = sample
    figure = plt.figure(figsize=(10, 10))

    figure.add_subplot(1, 2, 1)
    plt.title("Image")
    plt.axis("off")
    plt.imshow(image.numpy().transpose(1, 2, 0))

    figure.add_subplot(1, 2, 2)
    plt.title("Label")
    plt.axis("off")
    plt.imshow(label.numpy().transpose(1, 2, 0))

    plt.show()


def toOneHot(tensor, noClasses=32):
    H, W = tensor.shape
    oneHotTensor = torch.zeros((noClasses, H, W), dtype=torch.float64)

    for h in range(H):
        for w in range(W):
            oneHotTensor[int(tensor[h, w]), h, w] = 1

    return oneHotTensor


def generateMaskOld(image, values2classes):
    C, H, W = image.shape
    mask = torch.zeros((H, W), dtype=torch.int64)

    for h in range(H):
        for w in range(W):
            r, g, b = image[0, h, w], image[1, h, w], image[2, h, w]
            classLabel = values2classes[int(r)][int(g)][int(b)]
            mask[h, w] = classLabel

    return mask


def generateMask(image, values2classes):
    C, H, W = image.shape
    mask = torch.zeros((H, W), dtype=torch.int64)

    image_reshaped = image.view((C, H * W)).tolist()

    r, g, b = image_reshaped[0], image_reshaped[1], image_reshaped[2]

    mask = itemgetter(*tuple(zip(r, g, b)))(values2classes)

    mask = torch.tensor(mask, dtype=torch.int64).view((H, W))

    return mask


def unmask(mask, classes2values):
    H, W = mask.shape
    output = torch.zeros((3, H, W), dtype=torch.int64)

    for h in range(H):
        for w in range(W):
            classLabel = int(mask[h, w])
            r, g, b = classes2values[classLabel]
            output[0, h, w], output[1, h, w], output[2, h, w] = r, g, b

    return output


def segmentationLoss(output, target, values2classes):
    loss = 0

    crossEntropy = nn.CrossEntropyLoss()

    N, C, H, W = output.shape
    outputMask = output.reshape((N * C, -1)).transpose(1, 0)

    targetMask = generateMask(target.squeeze(0), values2classes)
    targetMask = targetMask.reshape((H * W)).to(target.device)

    loss = crossEntropy(outputMask, targetMask)

    return loss


def train_model(model, trainLoaders, lossFn, optimizer, values2classes, num_epochs=1, device="cpu",
                isSave=False, filename="unet-weights", verbose=True):
    since = datetime.datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    prev_train_loss = 0
    prev_val_loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, data in enumerate(tqdm(trainLoaders[phase], "Predicting ...")):
                torch.cuda.empty_cache()

                inputs = data["image"].to(device)
                labels = data["label"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # forward
                    outputs = model(inputs)

                    loss = lossFn(outputs, labels, values2classes)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item() * len(inputs))
                if verbose:
                    print(f' Iteration Loss: {loss.item() * len(inputs)}')

            epoch_loss = running_loss / len(trainLoaders[phase])

            if phase == "train":
                print(f"{phase} prev epoch Loss: {prev_train_loss}")
                prev_train_loss = epoch_loss

            if phase == "val":

                print(f"{phase} prev epoch Loss: {prev_val_loss}")
                prev_val_loss = epoch_loss

                # deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if isSave:
                        torch.save(model.state_dict(), f"trained/{filename}")

            print(f"{phase} current epoch Loss: {epoch_loss}")

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def evaluate_model(model, test_data, device="cpu"):
    since = datetime.datetime.now()

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for i, data in enumerate(tqdm(test_data, "Predicting ...")):
        inputs = data["image"].to(device)
        labels = data["labels"].to(device)

        # forward
        outputs = model(inputs)

    result = evalu(outputs, labels)

    print(f" Result : {result}")

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Evaluating complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return result
