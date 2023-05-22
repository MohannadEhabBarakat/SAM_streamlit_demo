
import requests
import base64, torch
from io import BytesIO 
import numpy as np
import json
from json import loads, dumps
import matplotlib.pyplot as plt
import cv2 

def resize_image(image):
  w, h = image.size
  print(w, h)
  max_w = 700
  max_h = 700

  if max_w<w and w>h:
    new_w = max_w
    new_h = int(h*(max_w/w))
    image = image.resize((new_w,new_h), Image.LANCZOS)

  elif max_h<h:
    new_h = max_h
    new_w = int(w*(max_h/h))
    image = image.resize((new_w,new_h), Image.LANCZOS)
  else:
    new_w, new_h = w, h

  return image, new_w, new_h

def run_sam_remote(objects, img, url, use_mask):
  headers = {
  'ngrok-skip-browser-warning': 'sdfsd',
  'Content-Type': 'application/json'
  }

  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()) 
  data = json.dumps({
      "image":img_str.decode()
  })
  r = requests.get(url=url+"/set_img", headers=headers, data=data)
  # print("r", r)
  # objects
  objects = objects.to_json()
  objects = loads(objects)
  objects = dumps(objects, indent=4)  

  data = json.dumps({
      "objects":objects,
      "use_mask":use_mask
  })
  r = requests.get(url=url+"/run_last_img", headers=headers, data=data)


  
  # extracting data in json format
  data = json.loads(r.content.decode())

  # print(data)
  return data['image']


def save_data_remote(objects, img, dataset_name, url, dino_arch, use_mask=False):
  headers = {
  'ngrok-skip-browser-warning': 'sdfsd',
  'Content-Type': 'application/json'
  }

  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()) 
  data = json.dumps({
      "image":img_str.decode()
  })
  r = requests.get(url=url+"/set_img", headers=headers, data=data)
  # print("r", r)
  # objects
  objects = objects.to_json()
  objects = loads(objects)
  objects = dumps(objects, indent=4)  

  data = json.dumps({
      "objects": objects,
      "name": dataset_name,
      "use_mask": use_mask,
      "dino_arch": dino_arch
  })
  requests.get(url=url+"/add_to_dataset", headers=headers, data=data)


def create_dataset(dataset_name, url):
  headers = {
  'ngrok-skip-browser-warning': 'sdfsd',
  'Content-Type': 'application/json'
  }

  data = json.dumps({
      "name": dataset_name
  })
  r = requests.get(url=url+"/create_dataset", headers=headers, data=data)
  data = json.loads(r.content.decode())
  return data["done"]



import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision
import sys

# Specify canvas parameters in application
url = st.sidebar.text_input("Enter URL:")

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "rect", "transform", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg", "jpeg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
use_mask = st.sidebar.checkbox("use last mask", True)

def reset(url):
  headers = {
    'ngrok-skip-browser-warning': 'sdfsd',
    'Content-Type': 'application/json'
    }
  r = requests.get(url=url+"/reset", headers=headers)
  st.info("Backend reseted")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


if st.sidebar.checkbox("Show image", False) and bg_image is not None:
  image = Image.open(bg_image)
  image, width, height = resize_image(image)
  # Create a canvas component
  # width = image.size[0]
  # height = image.size[1]
  canvas_result = st_canvas(
      fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
      stroke_width=stroke_width,
      stroke_color=stroke_color,
      background_color=bg_color,
      background_image=image if bg_image else None,
      update_streamlit=realtime_update,
      height=height,
      width=width,
      drawing_mode=drawing_mode,
      point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
      display_toolbar=st.sidebar.checkbox("Display toolbar", True),
      key="full_app",
  )

  if canvas_result.json_data is not None:
      objects = pd.json_normalize(canvas_result.json_data["objects"])
      for col in objects.select_dtypes(include=["object"]).columns:
          objects[col] = objects[col].astype("str")
      st.dataframe(objects)
      # st.write(str(type(objects)))

data = None
if st.sidebar.button('Run SAM'):
  data = None
  data = run_sam_remote(objects, image, url, use_mask)

if data is not None:
  masks = data
  fig = plt.figure(figsize=(10, 10))
  pil_image = image#.convert('RGB') 
  open_cv_image = np.array(pil_image) 
  open_cv_image = open_cv_image.copy() 
  plt.imshow(open_cv_image)
  for mask in masks:
      show_mask(np.array(mask), plt.gca(), random_color=True)
  plt.axis('off')
  st.pyplot(fig)

dataset_name = st.sidebar.text_input("Dataset Name")
data = None
if st.sidebar.button('Create Dataset'):
  data = None
  data = create_dataset(dataset_name, url)
  if data:
    st.info("Dataset created")
  else:
     st.info("Error")

dino_arch = st.sidebar.selectbox(
    "DINO Architecture:",
    ("vit_base_16", "vit_small_16"),
)

data = None
if st.sidebar.button('Save Data'):
  data = None
  save_data_remote(objects, image, dataset_name, url, dino_arch)
  st.info("All data saved successfully")

if st.sidebar.button('Reset backend'):
  reset(url)

