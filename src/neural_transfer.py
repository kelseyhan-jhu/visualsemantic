"""Neural transfer

This script supports neural transfer for texture and image synthesis.

This file contains the following functions and classes, adapted from the pytorch neural transfer tutorial (https://pytorch.org/tutorials/advanced/neural_style_tutorial.html):

    * image_loader
    * imshow
    * gram_matrix
    * ContentLoss
    * StyleLoss
    * Normalization
    * get_style_model_and_losses
    * run_style_transfer 
"""


from __future__ import print_function

import copy
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn  # import torch.nn as nn
from torch import optim  # import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms  # import torchvision.transforms as transforms
from torchvision import models  # import torchvision.models as models

import src.utils


def image_loader(image_name, imsize, device):
    """Loads an image from file system to device

    Parameters
    ----------
    image_name : str
        The file location of the image
    imsize : int
        The resize scale

    Returns
    -------
    torch tensor
        Image tensor on device

    """

    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device)  # image.to(device, torch.float)


def imshow(tensor, title=None):
    """Unloads a torch tensor image and plot

    Parameters
    ----------
    tensor : torch tensor
        Torch tensor to unload
    title : str
        The plot title

    """

    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class ContentLoss(nn.Module):
    """
    A torch nn module for content loss

    Attributes
    ----------
    target : torch tensor
        (From the original style transfer code:
        "we 'detach' the target content from the tree used
        to dynamically compute the gradient: this is a stated value,
        not a variable. Otherwise the forward method of the criterion
        will throw an error.")
    loss : torch tensor
        MSE loss between the input and the target features

    Methods
    -------
    forward(input)
        Computes the content loss
    """

    def __init__(
        self,
        target,
    ):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """Computes the gram matrix of a set of input features

    Parameters
    ----------
    input : torch tensor
        Input features of shape (a, b, c, d), where
        a = batch size(=1)
        b = number of feature maps
        (c, d) = dimensions of a feature map (N=c*d)

    Returns
    -------
    torch tensor
        Normalized gram matrix by dividing by the number of elements
        in each feature map
        (Normalization prevents a large feature dimension from making
        a larger impact during the gradient descent)
    """

    a, b, c, d = input.size()
    features = input.view(a * b, c * d)  # Resize to (b, N=c*d)
    G = torch.mm(features, features.t())  # compute the gram product

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """
    A torch nn module for style loss

    Attributes
    ----------
    target_feature : torch tensor
        Target feature
    loss : torch tensor
        MSE loss between the gram matrices of the input and the target features

    Methods
    -------
    forward(input)
        Computes the style loss
    """

    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """
    A module to normalize input image to easily put it in a nn.Sequential

    Attributes
    ----------
    mean : torch tensor
        Mean of the input image
    std : torch tensor
        Standard deviation of the input image

    Methods
    -------
    forward(img)
        Returns the normalized image tensor
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def init_weights(m):
    """Loads VGG-19 normalized weights of the model from DeepTextures

    Parameters
    ----------
    m : nn module
        VGG-19 net
    -------
    """

    model_weights = np.load(
        "/home/chan21/projects/visualsemantic/DeepTextures/Models/placeholder2.npy", allow_pickle=True
    ).item()
    model_weights = OrderedDict(model_weights)
    # caffe_eval = caffe_model.to(device).eval()
    conv_vals = [v for v in OrderedDict((model_weights)).values()]
    i = 0  # increment every time we see a conv
    for layer in m.children():
        if isinstance(layer, nn.Conv2d):
            name = "conv_{}".format(i)
            weight_source = torch.tensor(conv_vals[i]["weights"])
            bias_source = torch.tensor(conv_vals[i]["bias"]).squeeze()
            layer.weight.data.copy_(weight_source.clone().detach())
            layer.bias.data.copy_(bias_source.clone().detach())
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        # model.add_module(name, layer)
        # if isinstance(m, nn.Linear):
        #     torch.nn.init.xavier_uniform(m.weight)
        #     #m.bias.data.fill_(0.01)


def get_input_optimizer(input_img):
    """Input optimizer

    Parameters
    ----------
    input_img : torch tensor
        Input parameter that requires a gradient

    Returns
    -------
    optimizer

    """

    optimizer = optim.LBFGS([input_img])
    return optimizer


def get_style_model_and_losses(
    cnn,
    normalization,
    style_img,
    content_img,
    content_layers,
    style_layers,
):
    """Get model and content and style losses

    Parameters
    ----------
    cnn : nn module
        Base cnn model
    normalization: nn module
        Normalization module with pre-computed mean and std from input image
    style_img: torch tensor
        Target image tensor
    content_img: torch tensor
        Input image tensor
    content_layers: list
        List of cnn layers to compute content losses on
    style_layers: list
        List of cnn layesr to compute style losses on

    Returns
    -------
    model : nn module
        Style model to compute gradients on
    style_losses : list
        List of StyleLoss class
    content_losses : list
        List of ContentLoss class
    """

    content_losses = []
    style_losses = []

    # create a new nn.Sequential to put in content and style loss modules
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv (for content loss)
    j = 0  # increment every time we see a pooling (for style loss)

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)  # Turn off in-place ReLU
        elif isinstance(layer, nn.MaxPool2d):
            j += 1
            name = "pool_{}".format(j)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(j), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]  # trim off after the last content and style losses

    return model, style_losses, content_losses


def run_style_transfer(
    cnn,
    normalization,
    content_img,
    style_img,
    input_img,
    content_layers,
    style_layers,
    num_steps=500,
    style_weight=1000000000,
    content_weight=1,
):
    """Run the style transfer."""

    print("Building the style transfer model..")

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization,
        style_img,
        content_img,
        content_layers,
        style_layers,
    )

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print("Optimizing..")

    best_loss = [100000]  # update best loss to stop before overfitting
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                if loss.item() < best_loss[0]:
                    best_loss[0] = loss.item()
                    torch.save(
                        {
                            "run": run[0],
                            "style_score": style_score.item(),
                            "content_score": content_score.item(),
                            "loss": loss.item(),
                            "input_img": input_img,
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        "checkpoint.pt",
                    )

                print("run {}:".format(run))
                print(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


# if __name__ == "__main__":

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     parser = ArgumentParser()
#     parser.add_argument("--model", default="vgg_normalized")
#     parser.add_argument(
#         "--layer", nargs="+", default=["pool_1", "pool_2", "pool_3", "pool_4", "pool_5"]
#     )
#     parser.add_argument("--style_image", default="pebbles")
#     parser.add_argument("--input_image", default="random")
#     parser.add_argument("--style_weight", type=int, default=1000000000)
#     parser.add_argument("--content_weight", type=int, default=1)
#     parser.add_argument("--num_steps", type=int, default=500)

#     args = parser.parse_args()
#     model = args.model
#     layer = args.layer
#     style_image = args.style_image
#     input_image = args.input_image
#     style_weight = args.style_weight
#     content_weight = args.content_weight
#     num_steps = args.num_steps

#     IMSIZE = 256 if torch.cuda.is_available() else 128

#     loader = transforms.Compose([transforms.Resize(IMSIZE), transforms.ToTensor()])

#     normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

#     normalization = Normalization(normalization_mean, normalization_std).to(
#         device
#     )  # normalization module

#     if style_image.split("/")[0] == "things":
#         style_dataset = "things"
#         condition = style_image.split("/")[-1]
#         # conds = [c for c in src.utils.listdir(
#         #    "/data/chan21/semanticdimensionality/data/image/things")]
#         # for c in conds:
#         #    condition = c.split('/')[-1]
#         print(condition)
#         style_im = loader(
#             Image.open(
#                 src.utils.listdir(
#                     "/data/chan21/semanticdimensionality/data/image/" + style_image
#                 )[0]
#             )
#         ).unsqueeze(0)
#     else:
#         style_im = image_loader(style_image, IMSIZE)
#         style_im = style_im.to(device, torch.float)
#         style_dataset = None

#     # Set the input
#     if input_image == "random":
#         input_im = torch.randn(style_im.data.size(), device=device)

#     elif input_image.split("/")[0] == "things":
#         input_im = loader(
#             Image.open(
#                 src.utils.listdir(
#                     "/data/chan21/semanticdimensionality/data/image/" + input_image
#                 )[0]
#             )
#         ).unsqueeze(0)
#         input_im = input_im.to(device, torch.float)
#     else:
#         input_im = image_loader(input_image, IMSIZE)
#         input_im = input_im.to(device, torch.float)
#     content_im = input_im

#     # Set the model
#     if model == "vgg_normalized":
#         style_im = style_im[:, (2, 1, 0), :, :]
#         style_im = style_im.to(device, torch.float)

#         net = models.vgg19(pretrained=False).features.to(device).eval()
#         net.apply(init_weights)

#     elif model == "vgg":
#         net = models.vgg19(pretrained=True).features.to(device).eval()

#     # Add layers
#     style_layers_default = []

#     for l in layer:
#         style_layers_default.append(l)

#     content_layers_default = ["conv_1"]

#     output = run_style_transfer(
#         net,
#         normalization,
#         content_im,
#         style_im,
#         input_im,
#         content_layers=content_layers_default,
#         style_layers=style_layers_default,
#         num_steps=num_steps,
#         style_weight=style_weight,
#         content_weight=content_weight,
#     )

#     # Load the best image
#     checkpoint = torch.load("checkpoint.pt")
#     output_run = checkpoint['run']
#     output_loss = checkpoint['loss']
#     output = checkpoint['input_img']
#     with torch.no_grad():
#         output.clamp_(0, 1)

#     if model == "vgg_normalized":
#         output_BGR = output.requires_grad_(False)
#         output = output_BGR[:, (2, 1, 0), :, :]

#     plt.figure()
#     imshow(output, title="Output Image")

#     # sphinx_gallery_thumbnail_number = 4
#     plt.ioff()
#     plt.show()

#     filename = (
#         input_image.split("/")[-1]
#         + "_"
#         + style_image.split("/")[-1]
#         + "_"
#         + style_layers_default[-1]
#         + "_"
#         + str(output_run)
#         + ".npy"
#     )
#     print(filename)
#     np.save("../results/adversary/" + filename, output.cpu())
