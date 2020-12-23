import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilImage

#def np_to_tensor(array):
#    return torch.Tensor(array).permute(2,0,1).float()

def tensor_to_np_img(img):
    return (img.permute(1,2,0).cpu().numpy()*255.).astype('uint8')

def imresize(arr, sz):
    height, width = sz
    return np.array(arr_to_img(arr).resize((width, height), resample=pilImage.BILINEAR))

def min_max_scaler(x): 
    return (x-x.min())/(x.max()-x.min()+1e-8)

def arr_to_img(img, cmap=None):
    if type(img) == pilImage.Image:
        pass
    else:
        if cmap is not None:
            cm = plt.get_cmap(cmap)
            img = cm(img)[:,:,:3]
        img = pilImage.fromarray((min_max_scaler(img)*255).astype('uint8'))
    return img

# # Show linear layer activation function
# def show_1D_act(act):
#     '''Show activation visualizations of the l-th linear layer with figsize s*s'''
#     act_length = act.shape[-1]
#     act = act.numpy() if type(act)!=np.ndarray else act
#     act = np.array([[act],]*20).reshape(20,act.shape[-1]) # copy linear activation 20x for better visualization
#     act = arr_to_img(imresize(act, (20,400)))
#     #act = arr_to_img(act).resize((20,5),resample=pilImage.BILINEAR)
#     fig, ax = plt.subplots(figsize=(10,1))
#     ax.imshow(act, cmap='inferno')
#     ax.set_axis_off()