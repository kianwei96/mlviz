import gradio as gr
import numpy as np
from datetime import datetime
import time
import torch
import cv2
import utils

print('initializing')
# PAINTING_FPS = 1
CANVAS_DIM = (600, 800)
BRUSH_SIZE = 25
WINDOW_PADDING = 0.25
CLASSES = ['blue', 'red', 'white']

current_phase = 'drawing'
try_next_epoch = False
active_model = None
active_optimizer = None
current_class = CLASSES[0]
blank_img = 255*np.ones((*CANVAS_DIM,3), dtype=np.uint8)
training_img = np.random.randint(0, 255, (*CANVAS_DIM,3), dtype=np.uint8)
dummy_img = np.random.randint(0, 255, (*CANVAS_DIM,3), dtype=np.uint8)
neg_blue_points = np.zeros((0,2))
pos_red_points = np.zeros((0,2))
grid_pts = np.array(np.meshgrid(np.arange(CANVAS_DIM[1]), np.arange(CANVAS_DIM[0]))).reshape(2,-1).T
scaled_grid_pts = (grid_pts / (np.max(CANVAS_DIM) / 2)) - 1 

def _set_active_class(choice):
    global current_class, current_phase
    current_class = choice
    return gr.ImageEditor(layers=False, 
                    image_mode='RGB',
                    brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                    height=600, width=800,
                    container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                    visible=current_phase=='drawing',
                    interactive=True                    
                    )           

def _set_drawing_phase(input_canvas, training_canvas):
    global current_phase, training_img, try_next_epoch
    if not try_next_epoch:
        current_phase = 'drawing'
    input_canvas = gr.ImageEditor(layers=False, 
                            image_mode='RGB',
                            brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                            height=600, width=800,
                            container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                            visible=current_phase=='drawing',
                            interactive=True
                            )
    training_canvas = gr.Image(value=training_img, 
                                height=600, width=800,
                                visible=current_phase=='training', show_download_button=False, min_width=1000,
                                )    
    return input_canvas, training_canvas

def _set_training_phase(input_canvas, training_canvas, sample_density, sample_noise, model_toggle):
    global current_phase, neg_blue_points, pos_red_points, training_img, try_next_epoch

    if not try_next_epoch:
        neg_blue_points, pos_red_points = utils.draw_new_samples(input_canvas['composite'][:,:,:3], sample_density, sample_noise)
        if neg_blue_points.shape[0] > 0 and pos_red_points.shape[0] > 0:
            current_phase = 'training'
            print(f"{neg_blue_points.shape=}, {pos_red_points.shape=}")
            training_img = utils.render_data_points(training_img, neg_blue_points, pos_red_points)

    input_canvas = gr.ImageEditor(layers=False, 
                            image_mode='RGB',
                            brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                            height=600, width=800,
                            container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                            visible=current_phase=='drawing',
                            interactive=True
                            )
    training_canvas = gr.Image(value=training_img, 
                                height=600, width=800,
                                visible=current_phase=='training', show_download_button=False, min_width=1000,
                                )    
    model_toggle = gr.Button('Toggle Training', interactive=(current_phase=='training'))     
    return input_canvas, training_canvas, model_toggle

def _set_model_training_bool(model_layers, model_width, model_relu):
    global active_model, active_optimizer, try_next_epoch
    if try_next_epoch:
        try_next_epoch = False
        txt = 'Model Stopped!'
    else:
        active_model = utils.MLP(model_width, model_layers, activation='relu' if model_relu else 'none')
        active_optimizer = torch.optim.Adam(active_model.parameters(), lr=0.01)
        # print(f'{active_model=}')
        try_next_epoch = True
        txt = 'Model Instantiated!'
    return f'{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} {txt}'

def _try_train(training_canvas):
    global active_model, active_optimizer, try_next_epoch, neg_blue_points, pos_red_points, training_img, scaled_grid_pts, dummy_img
    latest_loss = '-'
    if try_next_epoch and (active_model is not None):
        # print('ready for next epoch!')
        scaled_neg_pts = (neg_blue_points / (np.max(CANVAS_DIM) / 2)) - 1 
        scaled_pos_pts = (pos_red_points / (np.max(CANVAS_DIM) / 2)) - 1 
        active_model, latest_loss = utils.train_one_epoch(active_model, active_optimizer, scaled_neg_pts, scaled_pos_pts, 1.0)
        latest_loss = str(latest_loss)
        heat_map = utils.model_inference(active_model, scaled_grid_pts).reshape(*CANVAS_DIM, 2)
        heat_map = (255 * heat_map).astype(np.uint8)
        dummy_img[:] = 0
        dummy_img[:,:,[2,0]] = heat_map
        # training_img[:,:,1] = 0
        # training_img[:,:,0] = (255 * heat_map[:,:,0]).astype(np.uint8)
        # training_img[:,:,2] = (255 * heat_map[:,:,1]).astype(np.uint8)
        training_img = utils.render_data_points(dummy_img, neg_blue_points, pos_red_points, flush_white=False)

    training_canvas = gr.Image(value=training_img, 
                                height=600, width=800,
                                visible=current_phase=='training', show_download_button=False, min_width=1000,
                                )            

    return training_canvas, latest_loss

def _set_white(input_canvas):
    global blank_img, current_class, BRUSH_SIZE, current_phase
    # cv2.imwrite('temp.png', input_canvas['composite'][:,:,:3])
    # print(f'seeing {input_canvas['composite'].sum()=}')
    if input_canvas['composite'].sum() == 0:
        input_canvas = gr.ImageEditor(value=blank_img,
                                layers=False, 
                                image_mode='RGB',
                                brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                                height=600, width=800,
                                container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                visible=current_phase=='drawing',
                                interactive=True
                                )
        print('reseting per tick')
    else:
        input_canvas = gr.ImageEditor(
                                layers=False, 
                                image_mode='RGB',
                                brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                                height=600, width=800,
                                container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                visible=current_phase=='drawing',
                                interactive=True
                                )        
    return input_canvas

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):    
            pass               
        with gr.Column(scale=4):
            class_btn = gr.Radio(CLASSES, value=CLASSES[0], label='Select Brush')        
            debug_text = gr.Textbox(render=False)
            with gr.Row():
                with gr.Column(scale=1):
                    sample_density = gr.Slider(minimum=1.0, maximum=100.0, label='Data:: Density of Sampled Data')
                with gr.Column(scale=1):
                    sample_noise = gr.Slider(minimum=0.0, maximum=50.0, label='Data:: % of Mislabeled Points')               

            set_drawing_phase = gr.Button('Back to Drawing')    
            generate_samples = gr.Button('Generate Data Points')

            input_canvas = gr.ImageEditor(value=blank_img,
                                    layers=False, 
                                    image_mode='RGB',
                                    brush=gr.Brush(colors=[current_class], default_size=BRUSH_SIZE), 
                                    height=600, width=800,
                                    container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                    visible=current_phase=='drawing',
                                    interactive=True
                                    )
            training_canvas = gr.Image(value=training_img, 
                                       height=600, width=800,
                                       visible=current_phase=='training', show_download_button=False, min_width=1000,
                                       )         
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_layers = gr.Slider(minimum=0, maximum=8, step=1, label='MLP:: Number of Hidden Layers')
                with gr.Column(scale=1):
                    model_width = gr.Slider(minimum=1, maximum=128, step=1, label='MLP:: Size of Each Layer')  
                with gr.Column(scale=1):
                    model_relu = gr.Checkbox(label='Use ReLU')   
                # with gr.Column(scale=1):
                #     model_save_output = gr.Checkbox(label='Save Visuals')                        
                with gr.Column(scale=1):
                    model_toggle = gr.Button('Toggle Training', interactive=(current_phase=='training'))                        

            debug_text = gr.Text('-', label='Training Info:')

            # output_canvas = gr.Image(height=600, width=800)
        with gr.Column(scale=1):  
            pass        

    # input_canvas.apply(fn=_clear_to_white,
    #                    inputs=input_canvas,
    #                    outputs=input_canvas)

    set_drawing_phase.click(fn=_set_drawing_phase,
                            inputs=[input_canvas, training_canvas],
                            outputs=[input_canvas, training_canvas])
    generate_samples.click(fn=_set_training_phase,
                            inputs=[input_canvas, training_canvas, sample_density, sample_noise, model_toggle],
                            outputs=[input_canvas, training_canvas, model_toggle])                     
    class_btn.change(fn=_set_active_class,
                     inputs=class_btn,
                     outputs=input_canvas)
    model_toggle.click(fn=_set_model_training_bool,
                       inputs=[model_layers, model_width, model_relu],
                       outputs=debug_text)
    
    # .then(fn=_save_single_mask, 
    #                                             inputs=input_canvas, 
    #                                             outputs=debug_text).then(fn=_update_debug_image, 
    #                                                                     inputs=None, 
    #                                                                     outputs=output_canvas)

    # workaround for when they accidentally bin things
    timer = gr.Timer(1)
    timer.tick(fn=_set_white, inputs=input_canvas, outputs=input_canvas)
    timer_train = gr.Timer(1/20)
    timer_train.tick(fn=_try_train, 
                     inputs=training_canvas, 
                     outputs=[training_canvas, debug_text])

demo.launch(share=True)