import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model


def train_tgt_classifier(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.eval()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))

    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            encoded = encoder(images).squeeze_()
            preds = classifier(encoded)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))
    return encoder, classifier
