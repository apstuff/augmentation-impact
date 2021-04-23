""" Original from https://github.com/jacobgil/pytorch-grad-cam but almost completely rewritten
"""
import torch
from torch import nn
from torch.autograd import Function
from src.utils import *
import copy
from copy import copy, deepcopy

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = input * positive_mask
        self.save_for_backward(positive_mask)
        return output

    @staticmethod
    def backward(self, grad_output):
        positive_mask_1 = self.saved_tensors[0]
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        return grad_output * positive_mask_1 * positive_mask_2


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.cuda = use_cuda
        self.model = model.cuda() if self.cuda else model
        self.model.eval()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input.cuda()) if self.cuda else self.forward(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        output[0][index].backward(retain_graph=True)
        return input.grad.cpu().data.numpy()[0]

    def get_gradient_act(self, x, target_index=None):
        gb = self(x, index=target_index)
        return deprocess_image(gb.transpose((1, 2, 0)))

class GradCam(nn.Module):
    def __init__(self, model, target_type='classification', layer_ids=[], use_cuda=False):
        super(GradCam, self).__init__()
        self.target_type = target_type
        self.cuda = use_cuda
        self.model = model.cuda if self.cuda else model
        self.model.eval()

        self.collect_hooks = True
        self.feature_activation = {}
        self.gradients = {}
        self.layer_ids = layer_ids
        print('register hooks for:')
        for name, module in dict([*self.model.named_modules()]).items():
            if name in layer_ids:
                print(name)
                module.register_forward_hook(self.save_activation(name))
                module.register_backward_hook(self.save_gradient(name))

    def save_gradient(self, layer_id):
        def fn(_, __, grad):
            if self.collect_hooks:
                self.gradients[layer_id] = grad[0]
            else:
                self.gradients[layer_id] = torch.empty(0)
        return fn

    def save_activation(self, layer_id):
        def fn(_, __, output):
            if self.collect_hooks:
                self.feature_activation[layer_id] = output
            else:
                self.feature_activation[layer_id] = torch.empty(0)

        return fn

    def __call__(self, input, index=None, feature_layer=None):
        feature_layer = self.layer_ids[0] if feature_layer is None else feature_layer
        output = self.model(input.cuda()) if self.cuda else self.model(input)
        feature_activation = self.feature_activation[feature_layer].cpu().data.numpy()[0,
                             :]  # 0 because pytorch always wants batches
        # temporary division between the 2 target types to easier develop the code
        if self.target_type == 'classification':
            if index is None:
                index = np.argmax(output.cpu().data.numpy())
            output = output[0][index]
        elif self.target_type == 'regression':
            pass
        else:
            assert 0, f'target_type {self.target_type} not known'

        output.backward(retain_graph=True)
        grads_val = self.gradients[feature_layer].cpu().data.numpy()  # get the gradients for the target features
        feature_impact = np.mean(grads_val, axis=(2, 3))[0, :]  # calculate average impact per feature

        cam = np.zeros(feature_activation.shape[1:], dtype=np.float32)
        for i, w in enumerate(feature_impact):
            cam += w * feature_activation[i, :, :]

        cam = min_max_scaler(cam)
        cam = imresize(cam, input.shape[2:])
        return cam

    def get_heatmap(self, x, target_index=None):
        mask = self(x, target_index)
        heatmap = arr_to_img(mask, cmap='inferno')
        return heatmap


class GuidedGradCam:
    def __init__(self, model, use_cuda, target_type, layer_ids):
        self.grad_cam = GradCam(model, target_type, layer_ids)
        self.gb_model = GuidedBackpropReLUModel(model=copy(deepcopy(model)), use_cuda=use_cuda)

    def __call__(self, x, target_index=None):
        '''Function to compute grad-cam, returns also cam heatmap and plain backwards-gradient
           If target_index is None, returns the map for the highest scoring category.
           Otherwise, targets the requested index.'''
        mask = self.grad_cam(x, target_index)
        heatmap = arr_to_img(mask, cmap='inferno')

        gb = self.gb_model(x, index=target_index)
        gb = gb.transpose((1, 2, 0))

        cam_mask = np.stack([mask, mask, mask], 2)
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        return heatmap, gb, cam_gb

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)



