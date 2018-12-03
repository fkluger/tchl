"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
from torch.nn import ReLU

from utilities.gradcam_misc_functions import *




from resnet.resnet_plus_lstm import resnet18rnn

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# from resnet.train import Config




class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # print([x.shape for x in grad_out])
            # print([x for x in grad_in])
            # print([x.shape for x in grad_in])
            self.gradients = grad_in[0]
            # print([x for x in grad_out])
            # print("hooked!", [type(x) for x in grad_in])
            # bla

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target):
        # Forward pass
        # model_output, _ = self.model(input_image.unsqueeze(0).unsqueeze(0))
        model_output, _ = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Backward pass
        model_output.backward(gradient=target)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """

        conv_output = self.model.forward_convs_single(x)
        conv_output.register_hook(self.save_gradient)

        return conv_output

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        offset = self.model.forward_fcs_single_offset(conv_output)
        return conv_output, offset


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, offset = self.extractor.forward_pass(input_image)
        if target is None:
            offset_target = torch.div(1., offset)
        else:
            offset_target = torch.from_numpy(target).float()

        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        offset.backward(gradient=offset_target, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam, offset_target


WIDTH = 1250
HEIGHT = 380

if __name__ == '__main__':
    # Get params

    pretrained_model = resnet18rnn()

    model_path = "/tnt/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/1/b32_181011-092706/model_best.ckpt"
    # model_path = "/tnt/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/1/b2_180921-223047/model_best.ckpt"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    pretrained_model.load_state_dict(checkpoint['state_dict'], strict=True)

    # image_path = "/tnt/data/scene_understanding/KITTI/rawdata/2011_10_03/2011_10_03_drive_0034_sync/image_02/data/0000003670.png"
    # image_path = "/tnt/data/scene_understanding/KITTI/rawdata/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000004.png"
    image_path = "/tnt/data/scene_understanding/KITTI/rawdata/2011_09_26/2011_09_26_drive_0061_sync//image_02/data/0000000000.png"
    img = Image.open(image_path).convert('RGB')

    pad_w = WIDTH - img.width
    pad_h = HEIGHT - img.height

    pad_w1 = int(pad_w / 2)
    pad_w2 = pad_w - pad_w1
    pad_h1 = int(pad_h / 2)
    pad_h2 = pad_h - pad_h1

    padded_image = np.pad(np.array(img), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')

    pixel_mean = [0.362365, 0.377767, 0.366744]

    tfs_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])

    prep_img = tfs_val(padded_image)
    prep_img = Variable(prep_img, requires_grad=True)

    # Grad cam
    grad_cam = GradCam(pretrained_model)
    # Generate cam mask
    cam_up, target_up = grad_cam.generate_cam(prep_img, np.array([1]))
    cam_down, target_down = grad_cam.generate_cam(prep_img, np.array([-1]))

    cam_up = cv2.resize(cam_up, (padded_image.shape[1], padded_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cam_down = cv2.resize(cam_down, (padded_image.shape[1], padded_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Save mask
    # img_with_map_up = class_activation_on_image(padded_image, cam_up)
    # img_with_map_down = class_activation_on_image(padded_image, cam_down)
    img_with_map = class_activation_on_image_combined(padded_image, cam_up.astype(np.float32)-cam_down.astype(np.float32))
    print('Grad cam completed')

    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img.unsqueeze(0).unsqueeze(0), target_up)
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    gradient_img = gradient_images(guided_grads)
    gray_gradient_img = gradient_images(grayscale_guided_grads).squeeze()
    img_with_grad = class_activation_on_image(padded_image, (gray_gradient_img*(cam_up.astype(np.float32)/255.)).astype(np.uint8))
    print('Guided backpropagation completed')

    # f1 = plt.figure()
    # ax1 = plt.subplot()
    # ax1.imshow(cam_up)
    # estm_mp = np.array([WIDTH / 2., (0.125 * (-1) + 0.5)*HEIGHT])
    # estm_nv = np.array([np.sin(0), np.cos(0)])
    # estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
    # print(estm_hl)
    # estm_h11 = np.cross(estm_hl, np.array([1, 0, 0]))
    # estm_h21 = np.cross(estm_hl, np.array([1, 0, -WIDTH]))
    # estm_h11 /= estm_h11[2]
    # estm_h21 /= estm_h21[2]
    # print(estm_h11, estm_h21)
    # ax1.plot([estm_h11[0], estm_h21[0]], [estm_h11[1], estm_h21[1]], '-', lw=2, c='r')
    #
    # f2 = plt.figure()
    # ax2 = plt.subplot()
    # ax2.imshow(cam_down)
    # estm_mp = np.array([WIDTH / 2., (-0.125 * (-1) + 0.5)*HEIGHT])
    # estm_nv = np.array([np.sin(0), np.cos(0)])
    # estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
    # print(estm_hl)
    # estm_h12 = np.cross(estm_hl, np.array([1, 0, 0]))
    # estm_h22 = np.cross(estm_hl, np.array([1, 0, -WIDTH]))
    # estm_h12 /= estm_h12[2]
    # estm_h22 /= estm_h22[2]
    # print(estm_h12, estm_h22)
    # ax2.plot([estm_h12[0], estm_h22[0]], [estm_h12[1], estm_h22[1]], '-', lw=2, c='m')

    plt.figure()
    plt.imshow(img_with_map)
    # plt.figure()
    # plt.imshow(gray_gradient_img)
    # plt.figure()
    # plt.imshow(img_with_grad)
    plt.show()
