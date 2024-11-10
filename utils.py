import numpy as np
import cv2
import copy
import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple feedforward network with or without relu activations.
    """
    def __init__(self, 
                 hidden_size: int,
                 n_hidden_layers: int, 
                 activation: str='none'
                 ):
        """Instantiate simple feedforward network with or without relu activations.

        Args:
            hidden_size (int): neurons per hidden layer.
            n_hidden_layers (int): number of hidden layers.
            activation (str, optional): _description_. Defaults to 'none'.
        """
        super(MLP, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        if n_hidden_layers > 0:
            self.layer_block = nn.ModuleList(
                [nn.Linear(2, hidden_size)] + 
                [nn.Linear(hidden_size, hidden_size) 
                for i in range(n_hidden_layers-1)])
            self.class_layer = nn.Linear(hidden_size, 2)
        else:
            self.class_layer = nn.Linear(2, 2)         
        if activation == 'relu':
            self.hidden_act = nn.ReLU()
        elif activation == 'none':
            self.hidden_act = nn.Identity()
        else:
            raise NotImplementedError('supplied activation string not valid! supported:none/relu!')
        
    def forward(self, x):
        if self.n_hidden_layers > 0:
            for layer in self.layer_block:
                x = layer(x)
                x = self.hidden_act(x)
        x = self.class_layer(x)
        return x

def train_one_epoch(model: MLP, 
                    optimizer: torch.optim, 
                    neg_features: np.ndarray, 
                    pos_features: np.ndarray, 
                    rescale_loss: float, 
                    subsample: float=1.0, 
                    DEVICE: str='cpu'
                    ):
    """performs one optimization pass on provided model, given positive and negative class datasets. model is assumed to be 2-output.

    Args:
        model (MLP): instantiated torch model.
        optimizer (torch.optim): torch optimizer hooked onto model parameters.
        neg_features (np.ndarray): nxf feature array.
        pos_features (np.ndarray): nxf feature array.
        rescale_loss (float): to optionally scale up loss computation (potentially speeding convergence).
        subsample (float, optional): to simulate stochastic batching, else gradient updates on full dataset. Defaults to 1.0.
        DEVICE (str, optional): device to run model training on. Defaults to 'cpu'.

    Returns:
        model (MLP): tuned torch model.
        loss (float): epoch loss.
    """

    features = np.vstack((neg_features, pos_features))
    labels = np.zeros((features.shape[0], 2))
    labels[:len(neg_features), 0] = 1
    labels[len(neg_features):, 1] = 1
    
    n_selected = np.random.choice(np.arange(len(labels)), int(subsample * len(labels)), replace=False).astype(int)
    features = features[n_selected]
    labels = labels[n_selected]

    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    pred = model(torch.as_tensor(features).to(torch.float).to(DEVICE))
    loss = rescale_loss * loss_fn(pred, torch.as_tensor(labels[:,:]).to(torch.float).to(DEVICE))              
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model, loss.item() / rescale_loss

def model_inference(model: MLP, 
                    features: np.ndarray, 
                    DEVICE: str='cpu'
                    ):
    """Evaluation pass on features by MLP model.

    Args:
        model (MLP): instantiated torch model.
        features (np.ndarray): nxf feature array.
        DEVICE (str, optional):  device to run model training on. Defaults to 'cpu'.

    Returns:
        predictions (np.ndarray): predicted classes (nxc), where c=2 for this demo, between (0-1)
    """
    model.eval()
    pred = nn.functional.sigmoid(model(torch.as_tensor(features).to(torch.float).to(DEVICE)))
    return pred.detach().cpu().numpy()   

def render_data_points(img: np.ndarray, 
                       neg_pts: np.ndarray, 
                       pos_pts: np.ndarray, 
                       flush_white: bool=True
                       ):
    """overlay or replace image with 2-class data points.

    Args:
        img (np.ndarray): hxwx3 (0-255) uint8 image array,
        neg_pts (np.ndarray): nx2 float or int array within image bounds (to be colored blue).
        pos_pts (np.ndarray): mx2 float or int array within image bounds (to be colored red).
        flush_white (bool, optional): replace image or overlay on current content. Defaults to True.

    Returns:
        img (np.ndarray): output with drawn points, of same shape as input img.
    """

    if flush_white:
        img[:] = 255
    for pt in neg_pts:
        cv2.circle(img, pt.astype(int), 6, (0, 0, 0), -1)
        cv2.circle(img, pt.astype(int), 4, (0, 0, 255), -1)
    for pt in pos_pts:
        cv2.circle(img, pt.astype(int), 6, (0, 0, 0), -1)        
        cv2.circle(img, pt.astype(int), 4, (255, 0, 0), -1)        
    return img

def draw_new_samples(class_image: np.ndarray, 
                     sample_density: float, 
                     sample_noise: float
                     ):
    """draw sample datapoints from a set of regions defined by a bi-colored mask.

    Args:
        class_image (np.ndarray): hxwx3 uint8 array, where white (255-255-255) indicates background, and (255-0-0) or (0-0-255) indicates possible sample regions for each of two classes.
        sample_density (float): density of points to sample from given areas (>0).
        sample_noise (float): percentage of points to flip classes (0-1.0).

    Returns:
        neg_samples_filtered (np.ndarray): nx2 samples within the image bounds for class 0.
        pos_samples_filtered (np.ndarray): nx2 samples within the image bounds for class 1.
    """

    class_image_copy = copy.deepcopy(class_image)
    img_dim = class_image_copy.shape[:2]

    white_mask = class_image_copy.sum(axis=2) == (255 * 3)
    class_image_copy[white_mask] = 0

    class_image_copy = cv2.resize(class_image_copy, dsize=(512,512))

    # check if there is space allocated per class
    _area_tally = class_image_copy.sum(axis=(0,1))
    if _area_tally[0] == 0 or _area_tally[2] == 0:
        return np.zeros((0,2)), np.zeros((0,2))
    
    _neg_mask_locs = np.array(np.where(class_image_copy[:,:,2]))[[1,0],:].T
    _pos_mask_locs = np.array(np.where(class_image_copy[:,:,0]))[[1,0],:].T

    neg_samples = 512 * np.random.random((int(sample_density*100), 2))
    masked = (neg_samples.astype(int)[:,None] == _neg_mask_locs).all(2).any(1)
    neg_samples_filtered = neg_samples[masked, :]
    pos_samples = 512 * np.random.random((int(sample_density*100), 2))
    masked = (pos_samples.astype(int)[:,None] == _pos_mask_locs).all(2).any(1)
    pos_samples_filtered = pos_samples[masked, :]

    neg_samples_filtered = img_dim[::-1] * (neg_samples_filtered / 512)
    pos_samples_filtered = img_dim[::-1] * (pos_samples_filtered / 512)    

    n_neg_flip = int(len(neg_samples_filtered) * (sample_noise / 100))
    n_pos_flip = int(len(pos_samples_filtered) * (sample_noise / 100))
    if n_neg_flip > 0:
        chunk = neg_samples_filtered[:n_neg_flip, :]
        neg_samples_filtered = neg_samples_filtered[n_neg_flip:, :]
        pos_samples_filtered = np.vstack((pos_samples_filtered, chunk))
    if n_pos_flip > 0:
        chunk = pos_samples_filtered[:n_pos_flip, :]
        pos_samples_filtered = pos_samples_filtered[n_pos_flip:, :]
        neg_samples_filtered = np.vstack((neg_samples_filtered, chunk))        

    return neg_samples_filtered, pos_samples_filtered
