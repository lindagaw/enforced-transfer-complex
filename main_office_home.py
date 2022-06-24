"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt_classifier, train_tgt_encoder_and_critic
from core import get_empirical_mean, get_empirical_covar, get_mahalanobis_dist, is_in_distribution

from utils import get_data_loader, init_model, init_random_seed, make_variable
from models import Discriminator

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

import pretty_errors
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    image_size = 299
    batch_size = 32
    # load dataset
    dataroot_art = "..//dcgan//datasets//OfficeHome//Art//"
    dataroot_clipart = "..//dcgan//datasets//OfficeHome//Clipart//"
    dataroot_product = "..//dcgan//datasets//OfficeHome/Product//"
    dataroot_real_world = "..//dcgan//datasets//OfficeHome//Real World//"

    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #AddGaussianNoise(0., 1.)
    ])

    dataset_art = datasets.ImageFolder(root=dataroot_art,
                               transform=transform)
    dataset_clipart = datasets.ImageFolder(root=dataroot_clipart,
                               transform=transform)
    dataset_product = datasets.ImageFolder(root=dataroot_product,
                               transform=transform)
    dataset_real_world = datasets.ImageFolder(root=dataroot_real_world,
                               transform=transform)

    train_set_art, test_set_art = torch.utils.data.random_split(dataset_art, [int(len(dataset_art)*0.8), len(dataset_art)-int(len(dataset_art)*0.8)])
    train_set_clipart, test_set_clipart = torch.utils.data.random_split(dataset_clipart, [int(len(dataset_clipart)*0.8), len(dataset_clipart)-int(len(dataset_clipart)*0.8)])
    train_set_product, test_set_product = torch.utils.data.random_split(dataset_product, [int(len(dataset_product)*0.8), len(dataset_product)-int(len(dataset_product)*0.8)])
    train_set_real_world, test_set_real_world = torch.utils.data.random_split(dataset_real_world, [int(len(dataset_real_world)*0.8), len(dataset_real_world)-int(len(dataset_real_world)*0.8)])

    dataloader_train_art = torch.utils.data.DataLoader(train_set_art, batch_size=batch_size, shuffle=True)
    dataloader_train_clipart = torch.utils.data.DataLoader(train_set_clipart, batch_size=batch_size, shuffle=True)
    dataloader_train_product = torch.utils.data.DataLoader(train_set_product, batch_size=batch_size, shuffle=True)
    dataloader_train_real_world = torch.utils.data.DataLoader(train_set_real_world, batch_size=batch_size, shuffle=True)

    dataloader_test_art = torch.utils.data.DataLoader(test_set_art, batch_size=batch_size, shuffle=True)
    dataloader_test_clipart = torch.utils.data.DataLoader(test_set_clipart, batch_size=batch_size, shuffle=True)
    dataloader_test_product = torch.utils.data.DataLoader(test_set_product, batch_size=batch_size, shuffle=True)
    dataloader_test_real_world = torch.utils.data.DataLoader(test_set_real_world batch_size=batch_size, shuffle=True)

    # art to clipart
    src_data_loader = dataloader_train_art
    tgt_data_loader = dataloader_train_product

    src_data_loader_eval = dataloader_test_art
    tgt_data_loader_eval = dataloader_test_product

    # load models
    inception = models.inception_v3(aux_logits=False, pretrained=True)
    critic = Discriminator()

    src_encoder = torch.nn.Sequential(*(list(inception.children())[:-1]))
    src_classifier = nn.Linear(2048, 31)

    tgt_encoder = torch.nn.Sequential(*(list(inception.children())[:-1]))
    tgt_classifier = nn.Linear(2048, 31)

    critic.cuda()
    src_encoder.cuda()
    tgt_encoder.cuda()
    src_classifier.cuda()
    tgt_classifier.cuda()

    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    critic, tgt_encoder = train_tgt_encoder_and_critic(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader)

    tgt_encoder, tgt_classifier = train_tgt_classifier(tgt_encoder, tgt_classifier, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print(">>> domain adaption without OOD <<<")
    eval_tgt(tgt_encoder, tgt_classifier, tgt_data_loader_eval)

    print(">>> domain adaptation with OOD <<<")

    src_empirical_mean = get_empirical_mean(src_encoder, tgt_encoder, critic, src_data_loader)
    tgt_empirical_mean = get_empirical_mean(src_encoder, tgt_encoder, critic, tgt_data_loader)
    src_empirical_covar = get_empirical_covar(src_encoder, tgt_encoder, critic, src_data_loader)
    tgt_empirical_covar = get_empirical_covar(src_encoder, tgt_encoder, critic, tgt_data_loader)

    src_avg_mahalanobis, src_std_mahalanobis = get_mahalanobis_dist(src_encoder, tgt_encoder, critic, src_data_loader)
    tgt_avg_mahalanobis, tgt_std_mahalanobis = get_mahalanobis_dist(src_encoder, tgt_encoder, critic, tgt_data_loader)

    y_pred = []
    y_true = []

    for step, (images, labels) in enumerate(tgt_data_loader_eval):
        for image, label in zip(images, labels):
            is_ind_with_src = \
                is_in_distribution(src_avg_mahalanobis, src_std_mahalanobis, \
                                    src_empirical_mean, src_empirical_covar, src_encoder, image)
            is_ind_with_tgt = \
                is_in_distribution(tgt_avg_mahalanobis, tgt_std_mahalanobis, \
                                    tgt_empirical_mean, tgt_empirical_covar, tgt_encoder, image)

            if is_ind_with_tgt:
                image = make_variable(torch.unsqueeze(image, 0))
                encoded = tgt_encoder(image).squeeze_()
                predicted = tgt_classifier(encoded).squeeze_().cpu()
                y_pred.append(torch.argmax(predicted).item())
                y_true.append(label.squeeze_().cpu())
            elif is_ind_with_src:
                image = make_variable(torch.unsqueeze(image, 0))
                encoded = src_encoder(image).squeeze_()
                predicted = src_classifier(encoded).squeeze_().cpu()
                y_pred.append(torch.argmax(predicted).item())
                y_true.append(label.squeeze_().cpu())

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(acc)
