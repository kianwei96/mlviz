import gradio as gr
import numpy as np
from datetime import datetime
import torch
import utils

#region shared global static variables
CANVAS_DIM = (600, 800)
BRUSH_SIZE = 25
WINDOW_PADDING = 0.25
CLASSES = ['blue', 'red', 'white']
grid_pts = np.array(np.meshgrid(np.arange(CANVAS_DIM[1]), np.arange(CANVAS_DIM[0]))).reshape(2,-1).T
scaled_grid_pts = (grid_pts / (np.max(CANVAS_DIM) / 2)) - 1 
#endregion

def _set_active_class(choice, state):
    """gradio:: set new gr.ImageEditor with newly stored color, depending on user's choice on Radio UI
    """
    state["current_class"] = choice
    return gr.ImageEditor(layers=False, 
                    image_mode='RGB',
                    brush=gr.Brush(colors=[state["current_class"]], default_size=BRUSH_SIZE), 
                    height=600, width=800,
                    container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                    visible=state["current_phase"]=='drawing',
                    interactive=True                    
                    ), state

def _set_drawing_phase(input_canvas, training_canvas, state):
    """gradio:: set session state to drawing, disabling and enabling correct UI elements
    """    
    if not state["try_next_epoch"]:
        state["current_phase"] = 'drawing'
    input_canvas = gr.ImageEditor(layers=False, 
                            image_mode='RGB',
                            brush=gr.Brush(colors=[state["current_class"]], default_size=BRUSH_SIZE), 
                            height=600, width=800,
                            container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                            visible=state["current_phase"]=='drawing',
                            interactive=True
                            )
    training_canvas = gr.Image(value=state["training_img"], 
                                height=600, width=800,
                                visible=state["current_phase"]=='training', show_download_button=False, min_width=1000,
                                )    
    return input_canvas, training_canvas, state

def _set_training_phase(input_canvas, training_canvas, sample_density, sample_noise, model_toggle, state):
    """gradio:: set session state to training, disabling and enabling correct UI elements
    """    
    if not state["try_next_epoch"]:
        neg_blue_points, pos_red_points = utils.draw_new_samples(input_canvas['composite'][:,:,:3], sample_density, sample_noise)
        state["neg_blue_points"] = neg_blue_points
        state["pos_red_points"] = pos_red_points    
        if state["neg_blue_points"].shape[0] > 0 and state["pos_red_points"].shape[0] > 0:
            state["current_phase"] = 'training'
            state["training_img"] = utils.render_data_points(state["training_img"], state["neg_blue_points"], state["pos_red_points"])

    input_canvas = gr.ImageEditor(layers=False, 
                            image_mode='RGB',
                            brush=gr.Brush(colors=[state["current_class"]], default_size=BRUSH_SIZE), 
                            height=600, width=800,
                            container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                            visible=state["current_phase"]=='drawing',
                            interactive=True
                            )
    utils.draw_new_samples()
    training_canvas = gr.Image(value=state["training_img"], 
                                height=600, width=800,
                                visible=state["current_phase"]=='training', show_download_button=False, min_width=1000,
                                )    
    model_toggle = gr.Button('Toggle Training', interactive=(state["current_phase"]=='training'))     
    return input_canvas, training_canvas, model_toggle, state

def _set_model_training_bool(model_layers, model_width, model_relu, state):
    """gradio:: set session flag for whether training should be happening, instantiate model if so.
    """        
    if state["try_next_epoch"]:
        state["try_next_epoch"] = False
        txt = 'Model Stopped!'
    else:
        state["active_model"] = utils.MLP(model_width, model_layers, activation='relu' if model_relu else 'none')
        state["active_optimizer"] = torch.optim.Adam(state["active_model"].parameters(), lr=0.01)
        state["try_next_epoch"] = True
        txt = 'Model Instantiated!'
    dt_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    return dt_now + ' ' + txt, state

def _try_train(training_canvas, state):
    """gradio:: regularly scheduled to train model if flag is set, if so it will also update image with live predictions.
    """
    latest_loss = '-'
    if state["try_next_epoch"] and (state["active_model"] is not None):
        scaled_neg_pts = (state["neg_blue_points"] / (np.max(CANVAS_DIM) / 2)) - 1 
        scaled_pos_pts = (state["pos_red_points"] / (np.max(CANVAS_DIM) / 2)) - 1 
        state["active_model"], latest_loss = utils.train_one_epoch(state["active_model"], 
                                                                   state["active_optimizer"], scaled_neg_pts, scaled_pos_pts, 1.0)
        latest_loss = str(latest_loss)
        heat_map = utils.model_inference(state["active_model"], scaled_grid_pts).reshape(*CANVAS_DIM, 2)
        heat_map = (255 * heat_map).astype(np.uint8)
        state["training_img"][:] = 0
        state["training_img"][:,:,[2,0]] = heat_map
        state["training_img"] = utils.render_data_points(state["training_img"], 
                                                         state["neg_blue_points"], state["pos_red_points"], flush_white=False)

    training_canvas = gr.Image(value=state["training_img"], 
                                height=600, width=800,
                                visible=state["current_phase"]=='training', show_download_button=False, min_width=1000,
                                )            

    return training_canvas, latest_loss, state

def _set_white(input_canvas, state):
    """gradio:: scheduled check to flush drawing canvas to white if erase canvas (inbuilt UI) is pressed.
    """    
    if input_canvas['composite'].sum() == 0:
        input_canvas = gr.ImageEditor(value=255*np.ones_like(state["training_img"]),
                                layers=False, 
                                image_mode='RGB',
                                brush=gr.Brush(colors=[state["current_class"]], default_size=BRUSH_SIZE), 
                                height=600, width=800,
                                container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                visible=state["current_phase"]=='drawing',
                                interactive=True
                                )
    else:
        input_canvas = gr.ImageEditor(
                                layers=False, 
                                image_mode='RGB',
                                brush=gr.Brush(colors=[state["current_class"]], default_size=BRUSH_SIZE), 
                                height=600, width=800,
                                container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                visible=state["current_phase"]=='drawing',
                                interactive=True
                                )        
    return input_canvas, state

with gr.Blocks() as demo:

    # session level states
    state = gr.State({
        "current_phase":'drawing',
        "current_class":CLASSES[0],
        "neg_blue_points":np.zeros((0,2)),
        "pos_red_points":np.zeros((0,2)),   
        "try_next_epoch":False,   
        "active_model":None,
        "active_optimizer":None,
        "training_img":255*np.ones((*CANVAS_DIM,3), dtype=np.uint8),
    })

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

            input_canvas = gr.ImageEditor(value=255*np.ones_like(state.value["training_img"]),
                                    layers=False, 
                                    image_mode='RGB',
                                    brush=gr.Brush(colors=[state.value["current_class"]], default_size=BRUSH_SIZE), 
                                    height=600, width=800,
                                    container=False, show_download_button=False, show_fullscreen_button=False, sources=(), transforms=(), eraser=False,
                                    visible=state.value["current_phase"]=='drawing',
                                    interactive=True
                                    )
            training_canvas = gr.Image(value=state.value["training_img"], 
                                       height=600, width=800,
                                       visible=state.value["current_phase"]=='training', show_download_button=False, min_width=1000,
                                       )         
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_layers = gr.Slider(minimum=0, maximum=8, step=1, label='MLP:: Number of Hidden Layers')
                with gr.Column(scale=1):
                    model_width = gr.Slider(minimum=1, maximum=128, step=1, label='MLP:: Size of Each Layer')  
                with gr.Column(scale=1):
                    model_relu = gr.Checkbox(label='Use ReLU')                        
                with gr.Column(scale=1):
                    model_toggle = gr.Button('Toggle Training', interactive=(state.value["current_phase"]=='training'))                        

            debug_text = gr.Text('-', label='Training Info:')
        with gr.Column(scale=1):  
            pass        

    # ui interactions
    set_drawing_phase.click(fn=_set_drawing_phase,
                            inputs=[input_canvas, training_canvas, state],
                            outputs=[input_canvas, training_canvas, state])
    generate_samples.click(fn=_set_training_phase,
                            inputs=[input_canvas, training_canvas, sample_density, sample_noise, model_toggle, state],
                            outputs=[input_canvas, training_canvas, model_toggle, state])                     
    class_btn.change(fn=_set_active_class,
                     inputs=[class_btn, state],
                     outputs=[input_canvas, state])
    model_toggle.click(fn=_set_model_training_bool,
                       inputs=[model_layers, model_width, model_relu, state],
                       outputs=[debug_text, state])
    
    # workaround for when they accidentally bin things
    timer = gr.Timer(2)
    timer.tick(fn=_set_white, inputs=[input_canvas, state], outputs=[input_canvas, state])
    timer_train = gr.Timer(1/45)
    timer_train.tick(fn=_try_train, 
                     inputs=[training_canvas, state], 
                     outputs=[training_canvas, debug_text, state])

demo.launch(share=False)