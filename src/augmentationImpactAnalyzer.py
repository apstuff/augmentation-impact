import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.utils import *


class AugmentationImpactAnalyzer():
    def __init__(self, img, model=None, cuda=False, add_output_act=False, restrict_classes=None, normalize=None,
                 guided_grad_cam=None, resize=None):
        self.img_orig = img.copy()
        self.img = img.copy()
        self.out_img = img.copy()
        self.x_orig = transforms.ToTensor()(img).unsqueeze(0)
        self.x_orig = self.x_orig.cuda() if cuda else self.x_orig
        self.model = model.cuda() if cuda else model
        self.add_output_act = add_output_act
        self.images = []
        self.activations = []
        self.restrict_classes = restrict_classes
        self.class_idx = [int(i) for i in restrict_classes.keys()] if restrict_classes is not None else None
        self.normalize = normalize if normalize is not None else lambda x: x
        self.cuda = cuda
        self.ggc = guided_grad_cam
        self.show_score = True
        self.out_score = None
        self.resize = resize

    def reset(self, new_img=None):
        self.img_orig = new_img if new_img is not None else self.img_orig
        self.img = self.img_orig.copy()
        self.x_orig = transforms.ToTensor()(new_img).unsqueeze(0) if new_img is not None else self.x_orig
        self.x_orig = self.x_orig.cuda() if self.cuda else self.x_orig
        self.images = []
        self.activations = []

    def tfm_brightness(self, brightness):
        x = TF.adjust_brightness(self.x, brightness)
        return x

    def tfm_centercrop(self, crop_size):
        x = transforms.CenterCrop(crop_size)(self.x)
        return x

    def tfm_perspective(self, perspective_w=None, perspective_h=None, perspective_d=0):
        width, height = self.x.shape[2:]
        w = perspective_w if perspective_w is not None else width
        h = perspective_h if perspective_h is not None else height
        d = perspective_d
        if (w == width) and (h == height) and (perspective_d == 0):
            # If no transformation should be made, skip!
            # Otherwise TF.perspective will modify the input slightly (e.g. for binary input)
            return self.x
        else:
            startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            endpoints = [[0, 0], [w, d], [w, h], [0, h]]
            return TF.perspective(self.x, startpoints, endpoints, pilImage.BILINEAR)

    def tfm_erase(self, erase_i=0, erase_j=0, erase_h=0, erase_w=0):
        return TF.erase(self.x, erase_i, erase_j, erase_h, erase_w, 0)

    def tfm_rotate(self, angle):
        return TF.rotate(self.x, angle=angle)

    def tfms(self, activation_localization=None, show=True, brightness=None, crop_size=None, rotate_ang=None, **args):
        '''Run function to perform several transformations.
           Args:
               activation_localization (str): {'none','cam','gradient','gradcam'}, default None
                                              Mode for visualizing the parts of the image that affected the model output most
           todo: add more docu
        '''
        self.x = self.x_orig
        # first run all augmentation methods
        # adjust brightness
        if brightness is not None:
            self.x = self.tfm_brightness(brightness)
        # center crop the image
        if crop_size is not None:
            self.x = self.tfm_centercrop(crop_size)
        # rotate the image
        if rotate_ang is not None:
            self.x = self.tfm_rotate(rotate_ang)
        # perspective distorition
        perspective_args = {k: v for k, v in args.items() if 'perspective' in k}
        if len(perspective_args):
            self.x = self.tfm_perspective(**perspective_args)
        # erase part of the image
        erase_args = {k: v for k, v in args.items() if 'erase' in k}
        if len(erase_args):
            self.x = self.tfm_erase(**erase_args)

        # then run activation localization
        if (activation_localization is not None) & (activation_localization != 'none'):
            if self.ggc is None:
                assert 0, "Please provide a guided_grad_cam model first"
            x_input = self.normalize(self.x).requires_grad_(True)
            if activation_localization == 'gradcam':
                heatmap = self.ggc.grad_cam.get_heatmap(x_input)
                # need to renormalize the image before adding it to another image
                act_loc = tensor_to_np_img(min_max_scaler(self.x[0].detach())) + np.float32(heatmap)
            elif activation_localization == 'guided-gradient':
                act_loc = self.ggc.gb_model.get_gradient_act(x_input)
            elif activation_localization == 'guided-gradcam':
                _, _, act_loc = self.ggc(x_input)
            else:
                assert 0, f'activation_localization "{activation_localization}" not known'

            out_img = act_loc
        else:
            out_img = self.x[0].detach().cpu().permute(1, 2, 0).numpy()

        if ((len(out_img.shape) == 3) and (out_img.shape[2] == 1)):
            self.out_img = arr_to_img(out_img, 'viridis')
        else:
            self.out_img = arr_to_img(out_img)

        if self.resize is not None:
            self.out_img = self.out_img.reshape(*self.resize)

        # then add model activation
        # Note: Here we use a different model as in the activation localization.
        # We coul use the same model and use reuse model output from forward pass above but it might add unnecessary complexity
        if self.add_output_act:
            out = self.model(self.normalize(self.x)).cpu().detach().numpy()
            if self.restrict_classes is not None:
                out = out[:, self.class_idx]
            self.activations.append(out)
            self.out_score = out.max()
            self.out_img = self.combine_activation_with_img(self.out_img, out)
        elif self.show_score:
            out = self.model(self.normalize(self.x)).cpu().detach().numpy()
            self.out_score = out.max()
        else:
            pass

        self.out_img = arr_to_img(self.out_img)
        self.images.append(self.out_img)

        if show:
            fig, ax = plt.subplots()
            if self.restrict_classes is not None:
                my_xticks = self.restrict_classes.values()
                nr_classes = len(self.restrict_classes)
                h, w = self.out_img.shape
                step_size = w / nr_classes
                ax.set_xticks(np.arange(nr_classes) * step_size + (step_size // 2))
                ax.set_xticklabels(my_xticks, rotation=90)
            if self.show_score:
                plt.title(self.out_score)
            ax.imshow(self.out_img)

    def combine_activation_with_img(self, img, act):
        # to make flatten activation better visible, the images will be streched by a factor of 20
        # and the width is adapted to the original image width
        shape = (20, img.shape[1])
        # combine transformed images and their layer activations
        act = arr_to_img(imresize(act, shape), 'inferno')
        # to work also with 2D images (e.g. timeseries)
        img_comb = np.vstack([img, act])
        return img_comb

    def create_gif(self, path):
        # transform to images, add backward loop and store as gif
        first_img, *imgs = self.images
        imgs += [img for img in imgs[::-1]]
        first_img.save(fp=path, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)