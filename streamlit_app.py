from PIL import Image
import numpy as np
import os
import time
import random
import cv2
import asyncio
import torch
from torch import nn
import streamlit as st
from streamlit_drawable_canvas import st_canvas

ALPHA = 2e-1
DEVICE = "cpu"
SPACE_DIM = 300

st.set_page_config(page_title="Control Panel", layout="wide")
column_left1, column_right1 = st.columns([1,1], vertical_alignment="top")

#region managing state initializations

if "drawn_canvas" not in st.session_state.keys():
    st.session_state.seed = 1
    st.session_state.drawn_canvas = np.zeros((SPACE_DIM,SPACE_DIM,3), dtype=np.uint8)
    st.session_state.training_bool = False
    st.session_state.model = None
    st.session_state.optimizer = None
    st.session_state.fixed_number_to_increment = 0
    st.session_state.epoch_count = 0
    st.session_state.epoch_loss = 0
    st.session_state.cached_img = None    
    st.session_state.t0 = time.time()
else:
    st.session_state.fixed_number_to_increment += 1
if "neg_samples" not in st.session_state.keys():
    st.session_state.neg_samples = np.zeros((0,2))
if "pos_samples" not in st.session_state.keys():
    st.session_state.pos_samples = np.zeros((0,2))

#endregion

#region helpers and callbacks

def fill_class_callback():
    if opt_data_class: 
        st.session_state.drawn_canvas[:,:,0] = 1
    else:
        st.session_state.drawn_canvas[:,:,2] = 1

def clear_class_callback():
    if opt_data_class: 
        st.session_state.drawn_canvas[:,:,0] = 0
    else:
        st.session_state.drawn_canvas[:,:,2] = 0

def overlay_axes_and_samples(drawing):
    axes = np.zeros_like(st.session_state.drawn_canvas)
    cv2.arrowedLine(axes, (SPACE_DIM//2, SPACE_DIM), (SPACE_DIM//2, 0), (255,255,255), thickness=1, tipLength=0.05)
    cv2.arrowedLine(axes, (0, SPACE_DIM//2), (SPACE_DIM, SPACE_DIM//2), (255,255,255), thickness=1, tipLength=0.05)
    base_img = np.clip((255 * drawing.astype(np.uint16)) + axes.astype(np.uint16), 0, 255).astype(np.uint8)
    for neg_px in st.session_state.neg_samples:
        cv2.circle(base_img, neg_px.astype(int), 6, (255, 255, 255), -1)
        cv2.circle(base_img, neg_px.astype(int), 4, (255, 0, 0), -1)        
    for pos_px in st.session_state.pos_samples:
        cv2.circle(base_img, pos_px.astype(int), 6, (255, 255, 255), -1)        
        cv2.circle(base_img, pos_px.astype(int), 4, (0, 0, 255), -1)   
    # print(f"{st.session_state.pos_samples.shape=}, {st.session_state.neg_samples.shape=}")
    return base_img

def draw_new_samples():

    # check if there is space allocated per class
    _area_tally = st.session_state.drawn_canvas.sum(axis=(0,1))
    if _area_tally[0] == 0 or _area_tally[2] == 0:
        return
    
    _neg_mask_locs = np.array(np.where(st.session_state.drawn_canvas[:,:,0]))[[1,0],:].T
    _pos_mask_locs = np.array(np.where(st.session_state.drawn_canvas[:,:,2]))[[1,0],:].T

    neg_samples = np.zeros((0,2))
    pos_samples = np.zeros((0,2))

    neg_samples = SPACE_DIM * np.random.random((opt_sample_density*100, 2))
    masked = (neg_samples.astype(int)[:,None] == _neg_mask_locs).all(2).any(1)
    if np.sum(masked) > 1:
        st.session_state.neg_samples = neg_samples[masked, :]
    pos_samples = SPACE_DIM * np.random.random((opt_sample_density*100, 2))
    masked = (pos_samples.astype(int)[:,None] == _pos_mask_locs).all(2).any(1)
    if np.sum(masked) > 1:
        st.session_state.pos_samples = pos_samples[masked, :]

def toggle_training():
    st.session_state.training_bool = not st.session_state.training_bool
    pass

class MLP(nn.Module):
    def __init__(self, hidden_size, n_hidden_layers, activation='none'):
        super(MLP, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        if n_hidden_layers > 0:
            self.layer_block = nn.ModuleList(
                [nn.Linear(2, hidden_size)] + 
                [nn.Linear(hidden_size, hidden_size) 
                for i in range(n_hidden_layers-1)])
            # for layer in self.layer_block:
            #     torch.nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
            #     torch.nn.init.uniform_(layer.bias, a=-0.01, b=0.01)
            self.class_layer = nn.Linear(hidden_size, 2)
        else:
            self.class_layer = nn.Linear(2, 2)
        # torch.nn.init.uniform_(self.class_layer.weight, a=-0.01, b=0.01)
        # torch.nn.init.uniform_(self.class_layer.bias, a=-0.01, b=0.01)            
        if activation == 'relu':
            self.hidden_act = nn.ReLU()
        elif activation == 'none':
            self.hidden_act = nn.Identity()
    def forward(self, x):
        if self.n_hidden_layers > 0:
            for layer in self.layer_block:
                x = layer(x)
                x = self.hidden_act(x)
        x = self.class_layer(x)
        return x

def train_one_epoch(model, optimizer, neg_samples, pos_samples, rescale_loss, subsample=1.0):

    features = np.vstack((neg_samples, pos_samples)) / SPACE_DIM
    features = 4 * (features - 0.5)
    labels = np.zeros((features.shape[0], 2))
    labels[:len(neg_samples), 0] = 1
    labels[len(neg_samples):, 1] = 1
    
    n_selected = np.random.choice(np.arange(len(labels)), int(subsample * len(labels)), replace=False).astype(int)
    features = features[n_selected]
    labels = labels[n_selected]
    # print('debug:', features.shape, labels.shape)

    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    pred = model(torch.as_tensor(features).to(torch.float).to(DEVICE))
    # print(f'{pred.shape=}, {labels.shape=}')
    loss = rescale_loss * loss_fn(pred, torch.as_tensor(labels[:,:]).to(torch.float).to(DEVICE))              
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model, loss.item() / rescale_loss

def model_inference(model, points):
    features = points / SPACE_DIM
    features = 4 * (features - 0.5)
    model.eval()
    pred = nn.functional.sigmoid(model(torch.as_tensor(features).to(torch.float).to(DEVICE)))
    return pred.detach().cpu().numpy()   

def overlay_axes_and_predictions(heat_map, neg_points, pos_points, neg_pred1, pos_pred1):
    
    heat_map = heat_map.copy()
    neg_pred = neg_pred1.copy()
    pos_pred = pos_pred1.copy()

    axes = np.zeros_like(st.session_state.drawn_canvas)
    cv2.arrowedLine(axes, (SPACE_DIM//2, SPACE_DIM), (SPACE_DIM//2, 0), (0,255,0), thickness=1, tipLength=0.05)
    cv2.arrowedLine(axes, (0, SPACE_DIM//2), (SPACE_DIM, SPACE_DIM//2), (0,255,0), thickness=1, tipLength=0.05)
    base_img = np.clip((heat_map.astype(np.uint16)) + axes.astype(np.uint16), 0, 255).astype(np.uint8)
    for neg_px, neg_p in zip(neg_points, neg_pred):
        cv2.circle(base_img, neg_px.astype(int), 6, (0  , 0, 0), -1)
        cv2.circle(base_img, neg_px.astype(int), 4, (255, 0, 0), -1)        
    for pos_px, pos_p in zip(pos_points, pos_pred):
        cv2.circle(base_img, pos_px.astype(int), 6, (0, 0,   0), -1)        
        cv2.circle(base_img, pos_px.astype(int), 4, (0, 0, 255), -1)   
    # print(f"{st.session_state.pos_samples.shape=}, {st.session_state.neg_samples.shape=}")
    return base_img

#endregion

#region user interface

# create user options for class regions
opt_data_class = st.sidebar.toggle("Blue/Red Class")    
opt_data_fill_class = st.sidebar.button("Fill Current Class", on_click=fill_class_callback)
opt_data_clear_class = st.sidebar.button("Clear Current Class", on_click=clear_class_callback)
# create user options for sample generation
opt_sample_density = st.sidebar.slider("Training Data Density", min_value=1, max_value=25, value=5) # per 0.1
opt_trigger_resample = st.sidebar.button("Draw New Samples", on_click=draw_new_samples)
# create user options for model training
opt_model_layers = st.sidebar.slider("Number of Hidden Layers" ,min_value=0, max_value=8, value=1)
opt_model_width = st.sidebar.slider("Number of Neurons per Hidden Layer", min_value=1, max_value=128, value=3)
opt_model_toggle = st.sidebar.button("Stop Training" if st.session_state.training_bool else "Start Training", on_click=toggle_training)
opt_model_relu = st.sidebar.checkbox("Use Non-linear Activation")
opt_model_rescale_loss = st.sidebar.slider("Loss Live Rescaling Ten-Power", min_value=0, max_value=5, value=1)

#endregion

with column_left1:
    # create a canvas component
    canvas_result = st_canvas(
        stroke_width=30,
        stroke_color=f"rgba(255, 0, 0, {ALPHA})" if opt_data_class else f"rgba(0, 0, 255, {ALPHA})",
        background_image=Image.fromarray(overlay_axes_and_samples(st.session_state.drawn_canvas)),
        initial_drawing = {"random": st.session_state.fixed_number_to_increment // 2},
        update_streamlit=True,
        width=SPACE_DIM,
        height=SPACE_DIM,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas_result",
        display_toolbar=False,
    )

with column_right1:
    pred_canvas = st.empty()
    ep_loss = st.empty()
    pass

# linking drawable-canvas to save blue-red channels independently
if canvas_result.image_data is not None:
    if canvas_result.image_data.shape[:2] == st.session_state.drawn_canvas.shape[:2]:
        if not opt_data_class: # blue
            st.session_state.drawn_canvas[:,:,2] += (canvas_result.image_data[:,:,2] > 0).astype(np.uint8)
        else: # red active
            st.session_state.drawn_canvas[:,:,0] += (canvas_result.image_data[:,:,0] > 0).astype(np.uint8)
        st.session_state.drawn_canvas = np.clip(st.session_state.drawn_canvas, 0, 1)

#asynchronously trains and updates visuals
async def trainer(session_state):
    grid = np.array(np.meshgrid(np.arange(SPACE_DIM),np.arange(SPACE_DIM))).reshape(2,-1).T
    while True:
        await asyncio.sleep(1/60)
        if session_state.training_bool and session_state.model is not None and session_state.optimizer is not None:
            session_state.model, session_state.epoch_loss = train_one_epoch(session_state.model, 
                                                                            session_state.optimizer,
                                                                            session_state.neg_samples, 
                                                                            session_state.pos_samples, 
                                                                            10**opt_model_rescale_loss)
            if session_state.epoch_count % 1 == 0:
                grid_pred = model_inference(session_state.model, grid).reshape(SPACE_DIM,SPACE_DIM,2)
                neg_preds = model_inference(session_state.model, session_state.neg_samples[:,[0,1]]).squeeze()
                pos_preds = model_inference(session_state.model, session_state.pos_samples[:,[0,1]]).squeeze()
                grid_img = np.zeros((SPACE_DIM,SPACE_DIM,3), dtype=np.uint8)
                grid_img[:,:,0] = (255 * grid_pred[:,:,0]).astype(np.uint8)
                grid_img[:,:,2] = (255 * grid_pred[:,:,1]).astype(np.uint8)
                session_state.cached_img = overlay_axes_and_predictions(grid_img,
                                                                        session_state.neg_samples,
                                                                        session_state.pos_samples,
                                                                        neg_preds,
                                                                        pos_preds)
            session_state.epoch_count += 1

        elif session_state.training_bool and session_state.model is None:
            random.seed(session_state.seed)
            os.environ['PYTHONHASHSEED'] = str(session_state.seed)
            np.random.seed(session_state.seed)
            session_state.epoch_count = 0
            session_state.model = MLP(opt_model_width, opt_model_layers, "relu" if opt_model_relu else "none")
            session_state.optimizer = torch.optim.Adam(session_state.model.parameters(), lr=0.005)
            st.session_state.t0 = time.time()
            print('training starting')  

        elif not session_state.training_bool:
            session_state.model = None

        if session_state.cached_img is not None:
            pred_canvas.image(session_state.cached_img)    
            ep_loss.text(f"eps: {session_state.epoch_count/(time.time()-st.session_state.t0):.2f}, epoch: {session_state.epoch_count}, loss: {session_state.epoch_loss:.5f}")                    

asyncio.run(trainer(st.session_state))