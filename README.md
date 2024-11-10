# mlviz

Data Generation |  Linear Model | Baseline Model 
:-------------------------:|:-------------------------:|:-------------------------:
![](./assets/datagen.gif) | ![](./assets/underfit.gif) | ![](./assets/okfit.gif)  

Simple Web-UI to demonstrate core machine learning concepts such as:
1. Importance of non-linear activation functions
2. Effect of additional "neurons" for single layer perceptron
3. Expressiveness with more layers
4. Relationships between:
    * Task complexity 
    * Data availability
    * Data quality
    * Model complexity

If in use in a educational institute/workplace setting, please drop me an email at kianwei@u.nus.edu, so that I can be aware of any new features to support (and also because I'm curious as to who actually finds this useful! :grinning:).

## Installation

```
conda create -n mlviz python=3.12
cd mlviz
pip install -r requirements.txt
```

## To Run

```
python app.py
```

## Demo

Apart from self-hosting/running locally by cloning this repository, you should be able to test the app hosted at [https://huggingface.co/spaces/kianwei96/mlviz](https://huggingface.co/spaces/kianwei96/mlviz) via HuggingFace Spaces. Note that it's only working on non-mobile devices for now. Drop me a message if it's down.

## Changelog

* (8-11-2024) 
    * Initial commit with Streamlit-based UI
    * Deployed in Streamlit Community Cloud
* (10-11-2024) 
    * Migrated from Streamlit to Gradio
    * Added data noise parameter
    * Cleaned up code
    * Deployed in HuggingFace Spaces
    * Mobile access is now broken -- To check component compatibility issues

## Todo

* ~~Clean up code~~
* Add functionality to save and load sketches
* Add feature engineering function and demo (e.g. radial encoding)
* ~~Reduce flickering when interacting with UI~~
* Debug and fix mobile crashing
