"""Adversarial adaptation to train target encoder."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def get_empirical_mean(src_encoder, tgt_encoder, critic, data_loader):

    feat_concat_norms = []

    for step, (images, labels) in enumerate(data_loader):
        feat_srcs = src_encoder(make_variable(images)).squeeze_()
        feat_tgts = tgt_encoder(make_variable(images)).squeeze_()

        for feat_src, feat_tgt in zip(feat_srcs, feat_tgts):
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat_norm = np.linalg.norm(feat_concat.cpu())
            feat_concat_norms.append(feat_concat_norm)

    return np.average(feat_concat_norms)

def get_empirical_covar(src_encoder, tgt_encoder, critic, data_loader):

    emprical_mean = get_empirical_mean(src_encoder, tgt_encoder, critic, data_loader)
    vals = []
    for step, (images, labels) in enumerate(data_loader):

        feat_srcs = src_encoder(make_variable(images)).squeeze_()
        feat_tgts = tgt_encoder(make_variable(images)).squeeze_()

        for feat_src, feat_tgt in zip(feat_srcs, feat_tgts):
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat_norm = np.linalg.norm(feat_concat.cpu())
            difference = feat_concat_norm - emprical_mean
            val = difference * difference
            vals.append(val)

    return np.average(vals)

def get_mahalanobis_dist(src_encoder, tgt_encoder, critic, data_loader):

    emprical_mean = get_empirical_mean(src_encoder, tgt_encoder, critic, data_loader)
    empirical_covar = get_empirical_covar(src_encoder, tgt_encoder, critic, data_loader)

    mahalanobis_dists = []

    for step, (images, labels) in enumerate(data_loader):

        feat_srcs = src_encoder(make_variable(images)).squeeze_()
        feat_tgts = tgt_encoder(make_variable(images)).squeeze_()

        for feat_src, feat_tgt in zip(feat_srcs, feat_tgts):
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat_norm = np.linalg.norm(feat_concat.cpu())
            difference = feat_concat_norm - emprical_mean
            val = difference * empirical_covar * difference
            mahalanobis_dists.append(val)

    avg_mahalanobis = np.average(mahalanobis_dists)
    std_mahalanobis = np.std(mahalanobis_dists)

    return avg_mahalanobis, std_mahalanobis

def is_in_distribution(avg_mahalanobis, std_mahalanobis, empirical_mean, empirical_covar, encoder, image):
    image = make_variable(torch.unsqueeze(image, 0))
    difference = np.linalg.norm(encoder(image).squeeze_().cpu()) - empirical_mean
    mahalanobis = difference * empirical_covar * difference

    if avg_mahalanobis - 0.20*std_mahalanobis < mahalanobis and mahalanobis < avg_mahalanobis + 0.20*std_mahalanobis:
        return True
    else:
        return False
