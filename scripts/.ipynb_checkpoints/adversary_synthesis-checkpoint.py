"""Synthesize adversary images

This script generates synthesized images with target style and input content.

For optimal results, the script loads the initial weights of a model from DeepTextures (https://github.com/leongatys/DeepTextures), converted from caffe to pytorch using MMdnn (https://github.com/microsoft/MMdnn).

This script contains the following functions:

    * main - the main function of the script
"""


from __future__ import print_function
from argparse import ArgumentParser
import numpy as np
import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

from src.utils import listdir
from src.neural_transfer import (
    image_loader,
    imshow,
    gram_matrix,
    ContentLoss,
    StyleLoss,
    Normalization,
    init_weights,
    get_input_optimizer,
    get_style_model_and_losses,
    run_style_transfer,
)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("--model", default="vgg_normalized")
    parser.add_argument(
        "--layer", nargs="+", default=["pool_1", "pool_2", "pool_3", "pool_4", "pool_5"]
    )
    parser.add_argument("--style_image", default="face")
    parser.add_argument("--style_index", type=int, default=999)
    parser.add_argument("--content_image", default="scene")
    parser.add_argument("--content_index", type=int, default=999)
    parser.add_argument("--style_weight", type=int, default=1000000000)
    parser.add_argument("--content_weight", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=500)

    args = parser.parse_args()

    model = args.model
    layer = args.layer
    style_image = args.style_image
    style_index = args.style_index
    content_image = args.content_image
    content_index = args.content_index
    style_weight = args.style_weight
    content_weight = args.content_weight
    num_steps = args.num_steps

    IMSIZE = 256 if torch.cuda.is_available() else 128

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    normalization = Normalization(normalization_mean, normalization_std).to(
        device
    )  # normalization module

    # Set the model
    if model == "vgg_normalized":
        net = models.vgg19(pretrained=False).features.to(device).eval()
        net.apply(init_weights)

    elif model == "vgg":
        net = models.vgg19(pretrained=True).features.to(device).eval()

    # Add layers
    style_layers_default = []

    for l in layer:
        style_layers_default.append(l)

    content_layers_default = ["conv_1"]

    # Set the style image
    style_path = "/home/chan21/projects/visualsemantic/stimuli/" + style_image
    style_images = listdir(style_path)
    content_path = "/home/chan21/projects/visualsemantic/stimuli/" + content_image
    content_images = listdir(content_path)
    
    if content_index != 999:
        c = content_images[content_index]
        content_im = image_loader(c, IMSIZE, device)
        content_im = content_im[:, (2, 1, 0), :, :]
        input_im = content_im  # filler for run_style_transfer
    
    if style_index != 999:
        s = style_images[style_index]
        style_im = image_loader(s, IMSIZE, device)
        if model == "vgg_normalized":
            style_im = style_im[:, (2, 1, 0), :, :]
            # style_im = style_im.to(device, torch.float)

        output = run_style_transfer(
            net,
            normalization,
            content_im,
            style_im,
            input_im,
            content_layers=content_layers_default,
            style_layers=style_layers_default,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
        )

        # Load the best image
        checkpoint = torch.load("checkpoint.pt")
        output_run = checkpoint["run"]
        output_loss = checkpoint["loss"]
        output_style = checkpoint["style_score"]
        output_content = checkpoint["content_score"]
        output = checkpoint["input_img"]
        with torch.no_grad():
            output.clamp_(0, 1)

        if model == "vgg_normalized":
            output_BGR = output.requires_grad_(False)
            output = output_BGR[:, (2, 1, 0), :, :]

        filename = (
            s.split("/")[-1]
            + "_"
            + c.split("/")[-1]
            + "_"
            + style_layers_default[-1]
            + ".pt"
        )

        torch.save(
            {
                "run": output_run,
                "loss": output_loss,
                "style_score": output_style,
                "content_score": output_content,
                "input_img": output,
            },
            "/home/chan21/projects/visualsemantic/results/adversary/" +
            filename,
        )
        
    else:
        for s in style_images:
            style_im = image_loader(s, IMSIZE, device)
            if model == "vgg_normalized":
                style_im = style_im[:, (2, 1, 0), :, :]
                # style_im = style_im.to(device, torch.float)

            output = run_style_transfer(
                net,
                normalization,
                content_im,
                style_im,
                input_im,
                content_layers=content_layers_default,
                style_layers=style_layers_default,
                num_steps=num_steps,
                style_weight=style_weight,
                content_weight=content_weight,
            )

            # Load the best image
            checkpoint = torch.load("checkpoint.pt")
            output_run = checkpoint["run"]
            output_loss = checkpoint["loss"]
            output_style = checkpoint["style_score"]
            output_content = checkpoint["content_score"]
            output = checkpoint["input_img"]
            with torch.no_grad():
                output.clamp_(0, 1)

            if model == "vgg_normalized":
                output_BGR = output.requires_grad_(False)
                output = output_BGR[:, (2, 1, 0), :, :]

            filename = (
                s.split("/")[-1]
                + "_"
                + c.split("/")[-1]
                + "_"
                + style_layers_default[-1]
                + ".pt"
            )

            torch.save(
                {
                    "run": output_run,
                    "loss": output_loss,
                    "style_score": output_style,
                    "content_score": output_content,
                    "input_img": output,
                },
                "/home/chan21/projects/visualsemantic/results/adversary/" +
                filename,
            )
        
#         filename = (
#             s.split("/")[-1]
#             + "_"
#             + c.split("/")[-1]
#             + "_"
#             + style_layers_default[-1]
#             + "_"
#             + str(output_run)
#             + "_"
#             + str(output_loss)
#             + ".npy"
#         )
#         print(filename)
#         np.save("/home/chan21/projects/visualsemantic/results/adversary/" + filename, output.cpu())


# from __future__ import print_function

# import torch
# from torch import nn #import torch.nn as nn
# from torch import optim #import torch.optim as optim
# import torch.nn.functional as F

# from PIL import Image
# import matplotlib.pyplot as plt

# from torchvision import transforms #import torchvision.transforms as transforms
# rom torchvision import models #import torchvision.models as models
# import numpy as np
# import copy
# from collections import OrderedDict

# import src.utils

# from argparse import ArgumentParser


# def image_loader(image_name):
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     return image.to(device) #image.to(device, torch.float)

# def imshow(tensor, title=None):
#     unloader = transforms.ToPILImage()
#     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# class ContentLoss(nn.Module):
#     def __init__(
#         self,
#         target,
#     ):
#         super(ContentLoss, self).__init__()
#         # we 'detach' the target content from the tree used
#         # to dynamically compute the gradient: this is a stated value,
#         # not a variable. Otherwise the forward method of the criterion
#         # will throw an error.
#         self.target = target.detach()

#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input


# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)

#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

#     G = torch.mm(features, features.t())  # compute the gram product

#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)


# class StyleLoss(nn.Module):
#     def __init__(self, target_feature):
#         super(StyleLoss, self).__init__()
#         self.target = gram_matrix(target_feature).detach()

#     def forward(self, input):
#         G = gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input


# # create a module to normalize input image so we can easily put it in a
# # nn.Sequential
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)

#     def forward(self, img):
#         # normalize img
#         return (img - self.mean) / self.std


# def init_weights(m):
#     model_weights = np.load(
#         "../../visualsemantic/DeepTextures/Models/placeholder2.npy", allow_pickle=True
#     ).item()
#     model_weights = OrderedDict(model_weights)
#     # caffe_eval = caffe_model.to(device).eval()
#     conv_vals = [v for v in OrderedDict((model_weights)).values()]
#     i = 0  # increment every time we see a conv
#     for layer in m.children():
#         if isinstance(layer, nn.Conv2d):
#             name = "conv_{}".format(i)
#             weight_source = torch.tensor(conv_vals[i]["weights"])
#             bias_source = torch.tensor(conv_vals[i]["bias"]).squeeze()
#             layer.weight.data.copy_(weight_source.clone().detach())
#             layer.bias.data.copy_(bias_source.clone().detach())
#             i += 1
#         elif isinstance(layer, nn.ReLU):
#             name = "relu_{}".format(i)
#             # The in-place version doesn't play very nicely with the ContentLoss
#             # and StyleLoss we insert below. So we replace with out-of-place
#             # ones here.
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             name = "pool_{}".format(i)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = "bn_{}".format(i)
#         else:
#             raise RuntimeError(
#                 "Unrecognized layer: {}".format(layer.__class__.__name__)
#             )

#         # model.add_module(name, layer)
#         # if isinstance(m, nn.Linear):
#         #     torch.nn.init.xavier_uniform(m.weight)
#         #     #m.bias.data.fill_(0.01)


# def get_style_model_and_losses(
#     cnn,
#     normalization_mean,
#     normalization_std,
#     style_img,
#     content_img,
#     content_layers,
#     style_layers,
# ):
#     # normalization module
#     normalization = Normalization(normalization_mean, normalization_std).to(device)

#     # just in order to have an iterable access to or list of content/syle
#     # losses
#     content_losses = []
#     style_losses = []

#     # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
#     # to put in modules that are supposed to be activated sequentially
#     model = nn.Sequential(normalization)

#     i = 0  # increment every time we see a conv
#     j = 0  # increment every time we see a pooling

#     for layer in cnn.children():
#         if isinstance(layer, nn.Conv2d):
#             i += 1
#             name = "conv_{}".format(i)
#         elif isinstance(layer, nn.ReLU):
#             name = "relu_{}".format(i)
#             # The in-place version doesn't play very nicely with the ContentLoss
#             # and StyleLoss we insert below. So we replace with out-of-place
#             # ones here.
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             j += 1
#             name = "pool_{}".format(j)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = "bn_{}".format(i)
#         else:
#             raise RuntimeError(
#                 "Unrecognized layer: {}".format(layer.__class__.__name__)
#             )

#         model.add_module(name, layer)

#         if name in content_layers:
#             # add content loss:
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)

#         if name in style_layers:
#             # add style loss:
#             target_feature = model(style_img).detach()
#             style_loss = StyleLoss(target_feature)
#             model.add_module("style_loss_{}".format(j), style_loss)
#             style_losses.append(style_loss)

#     # now we trim off the layers after the last content and style losses
#     for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break

#     model = model[: (i + 1)]

#     print(model)

#     return model, style_losses, content_losses


# def get_input_optimizer(input_img):
#     # this line to show that input is a parameter that requires a gradient
#     optimizer = optim.LBFGS([input_img])
#     return optimizer


# def run_style_transfer(
#     cnn,
#     normalization_mean,
#     normalization_std,
#     content_img,
#     style_img,
#     input_img,
#     content_layers,
#     style_layers,
#     num_steps=500,
#     style_weight=1000000000,
#     content_weight=1,
# ):
#     """Run the style transfer."""
#     print("Building the style transfer model..")
#     model, style_losses, content_losses = get_style_model_and_losses(
#         cnn,
#         normalization_mean,
#         normalization_std,
#         style_img,
#         content_img,
#         content_layers,
#         style_layers,
#     )

#     # We want to optimize the input and not the model parameters so we
#     # update all the requires_grad fields accordingly
#     input_img.requires_grad_(True)
#     model.requires_grad_(False)

#     optimizer = get_input_optimizer(input_img)

#     print("Optimizing..")
#     run = [0]
#     while run[0] <= num_steps:

#         def closure():
#             # correct the values of updated input image
#             with torch.no_grad():
#                 input_img.clamp_(0, 1)

#             optimizer.zero_grad()
#             model(input_img)
#             style_score = 0
#             content_score = 0

#             for sl in style_losses:
#                 style_score += sl.loss
#             for cl in content_losses:
#                 content_score += cl.loss

#             style_score *= style_weight
#             content_score *= content_weight

#             loss = style_score + content_score
#             loss.backward()

#             run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print("Style Loss : {:4f}".format(style_score.item()))
#                 print(
#                     "Style Loss : {:4f} Content Loss: {:4f}".format(
#                         style_score.item(), content_score.item()
#                     )
#                 )
#                 print()

#             return style_score + content_score

#         optimizer.step(closure)

#     # a last correction...
#     with torch.no_grad():
#         input_img.clamp_(0, 1)

#     return input_img


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

#     # Set the style image
#     imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

#     loader = transforms.Compose(
#         [transforms.Resize(imsize), transforms.ToTensor()]  # scale imported image
#     )  # transform it into a torch tensor

#     normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

#     if style_image.split("/")[0] == "things":
#         style_dataset = "things"
#         condition = style_image.split("/")[-1]
#         # conds = [c for c in src.utils.listdir(
#         #    "/data/chan21/semanticdimensionality/data/image/things")]
#         # for c in conds:
#         #    condition = c.split('/')[-1]
#         print(condition)
#         style_img = loader(
#             Image.open(
#                 src.utils.listdir(
#                     "/data/chan21/semanticdimensionality/data/image/" + style_image
#                 )[0]
#             )
#         ).unsqueeze(0)
#     else:
#         style_img = image_loader(style_image)
#         style_img = style_img.to(device, torch.float)
#         style_dataset = None

#     # Set the input
#     if input_image == "random":
#         input_img = torch.randn(style_img.data.size(), device=device)

#     elif input_image.split("/")[0] == "things":
#         input_img = loader(
#             Image.open(
#                 src.utils.listdir(
#                     "/data/chan21/semanticdimensionality/data/image/" + input_image
#                 )[0]
#             )
#         ).unsqueeze(0)
#         input_img = input_img.to(device, torch.float)
#     else:
#         input_img = image_loader(input_image)
#         input_img = input_img.to(device, torch.float)
#     content_img = input_img

#     # Set the model
#     if model == "vgg_normalized":
#         style_img = style_img[:, (2, 1, 0), :, :]
#         style_img = style_img.to(device, torch.float)

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
#         normalization_mean,
#         normalization_std,
#         content_img,
#         style_img,
#         input_img,
#         content_layers=content_layers_default,
#         style_layers=style_layers_default,
#         num_steps=num_steps,
#         style_weight=style_weight,
#         content_weight=content_weight,
#     )

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
#         + str(num_steps)
#         + ".npy"
#     )
#     np.save("../results/adversary/" + filename, output.cpu())
