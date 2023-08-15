from flask import Flask, redirect, url_for, render_template, request, flash, send_from_directory
import tensorflow as tf
from tensorflow import keras
import numpy as np
from werkzeug.utils import secure_filename
import os

import numpy as np
import PIL
import tensorflow as tf
from pathlib import Path
import os
import cv2
import glob

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

import imutils
import pyarabic.trans
import glob, os

spaces = []
counter = 0
myfile = ""

UPLOAD_FOLDER = './templates/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

class_names = ['aaalef', 'alef', 'alef_hamza', 'alef_waw', 'alef_yaa', 'baa', 'dal', 'fa', 'gem', 'ghain', 'ha', 'hha', 'kaf', 'kha', 'lam', 'meem', 'noon', 'qaaf', 'thaa', 'thad', 'tta', 'ttha', 'waw', 'ya']

def allowed_file(filename):     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

my_model = tf.keras.models.load_model('braille_arabic_model')

app = Flask(__name__, static_folder="./static/")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def one_contour_each_line(contours,tolerance=4) :

  raw_contours = []
  last_y = None
  for contour in contours:
    y = cv2.boundingRect(contour)[1]
    if last_y is None or abs(y - last_y) > tolerance:
      raw_contours.append(contour)
      last_y = y
  return raw_contours

def one_contour_each_column(contours,tolerance=4) :
  column_contours = []

    # Loop over the sorted contours and add them to the final list if they have a unique x-coordinate
  for contour in contours:
          x, _, w, _ = cv2.boundingRect(contour)
          is_unique = True
          for column_contour in column_contours:
              final_x, _, final_w, _ = cv2.boundingRect(column_contour)
              if abs(x - final_x) <= tolerance:
                  is_unique = False
                  break
          if is_unique:
              column_contours.append(contour)

  column_contours = sorted(column_contours, key=lambda contour: cv2.boundingRect(contour)[0])
  return column_contours

def split_each_line(raw_contours) :
 
  raw_coords = []
  max_y_coords = [max(cv2.boundingRect(contour)[1] for contour in sublist) for sublist in raw_contours]
  
  for i in range(1,len(raw_contours),3) :

        y = cv2.boundingRect(raw_contours[i])[1]
        #cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 255), 1)
        raw_coords.append(y)

        if i+2 < len(max_y_coords) :
          raw_coords.append(max_y_coords[i+2])
          
  return raw_coords

def  split_each_column(column_contours) :
  column_coords = []
  x1, y1, w1, h1 = cv2.boundingRect(column_contours[1])
  width = w1
  i=1
  while i < len(column_contours) :

        x = cv2.boundingRect(column_contours[i])[0]
        if i+1 < len(column_contours) :
          x_next = cv2.boundingRect(column_contours[i+1])[0]
        #cv2.line(img, (x, 0), (x,img.shape[0]), (0, 0, 255), 1)
        column_coords.append(x)
        if x_next - x >= 2*width :
          i=i-1
        i=i+2
  #column_coords.append(cv2.boundingRect(column_contours[len(column_contours)-1])[0])
  return column_coords

def draw_grid(img,raw_cords,column_coords) :
  for raw in raw_cords :
    cv2.line(img, (0, raw), (img.shape[1], raw), (0, 0, 255), 1)
  for column in column_coords :
    cv2.line(img, (column, 0), (column, img.shape[0]), (0, 255, 0), 1)

def braille_cells(raw_coords,column_coords,img) :
  cells = []
  for i in range(0, len(raw_coords)-1 , 1):
      for j in range(0, len(column_coords) , 1):
          if i+1 < len(raw_coords) and j+1 < len(column_coords) :
            cell = img[raw_coords[i]:raw_coords[i+1] , column_coords[j]:column_coords[j+1]]
          if j == len(column_coords)-1 :
            cell = img[raw_coords[i]:raw_coords[i+1] , column_coords[j]:]
          cells.append(cell)
  return cells

def prepare_image(img_path) :
  img = cv2.imread(os.path.join(r"./templates/images/", img_path), cv2.IMREAD_GRAYSCALE)
  img = cv2.GaussianBlur(img, (5, 5), 0)
  _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])
  return img,contours

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global myfile
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            myfile = filename


            return redirect(url_for('braille_result'))
    return '''
    <!doctype html>
    <head>
        <link rel="icon" href="/static/favicon.ico">
        <style>
            @import url("https://fonts.googleapis.com/css?family=Fjalla+One&display=swap");

            * {
                margin: 0;
                padding: 0; }

            body {
                background: url('/static/background.jpg') center center no-repeat;
                background-size: cover;
                width: 100vw;
                height: 100vh;
                display: grid;
                align-items: center;
                justify-items: center; 
            }

            h1{
                user-select: none;
                font-family: "Helvetica", Times, serif;
                font-weight: 1000;               
            }

          .contact-us {
            background: #f8f4e5;
            padding: 50px 100px;
            border: 2px solid black;
            box-shadow: 15px 15px 1px #ffa580, 15px 15px 1px 2px black;
            }

          .back{
            background-color: rgb(247,126,65);
            border-radius: 10px;
            position: relative;
            top: 70px; 
            padding: 0px 10px; 
            border-style: solid;
            border-color: black;
            border-width: thick;

          }

          input {
            cursor: pointer;
            display: block;
            width: 100%;
            font-size: 14pt;
            line-height: 28pt;
            font-family: "Fjalla One";
            margin-bottom: 28pt;
            border: none;
            border-bottom: 5px solid black;
            background: #f8f4e5;
            min-width: 250px;
            padding-left: 5px;
            outline: none;
            color: black; }

          input:focus {
            border-bottom: 5px solid #ffa580; }

          button {
            display: block;
            margin: 0 auto;
            line-height: 28pt;
            padding: 0 20px;
            background: #ffa580;
            letter-spacing: 2px;
            transition: .2s all ease-in-out;
            outline: none;
            border: 1px solid black;
            box-shadow: 3px 3px 1px 1px #95a4ff, 3px 3px 1px 2px black; }
            button:hover {
              background: black;
              color: white;
              border: 1px solid black; }

          .logo{
            user-select: none;
            position: relative;
            top: 40px;
            width: 200px;
          }

          ::selection {
            background: #ffc8ff; }

          input:-webkit-autofill,
          input:-webkit-autofill:hover,
          input:-webkit-autofill:focus {
            border-bottom: 5px solid #95a4ff;
            -webkit-text-fill-color: #2A293E;
            -webkit-box-shadow: 0 0 0px 1000px #f8f4e5 inset;
            transition: background-color 5000s ease-in-out 0s; }
        </style>
        <title>Braille Project</title>
    </head>
    <div class="back">
        <h1>UPLOAD YOUR BRAILLE IMAGE</h1>
    </div>
    <img src="static/braille.png" class="logo"/>
    <form method="POST" enctype="multipart/form-data">
      <input type=file name=file accept=".png,.jpg,.jpeg">
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/braille_result/', methods=['GET', 'POST'])  
def braille_result(): 
  global myfile
  global counter
  string = "Error"
  i = 0
  text = []

  img, contours = prepare_image(myfile)

  raw_contours = [] 
  column_contours = [] 
  raw_coords = []  
  column_coords = []  
  cells = []
  raw_contours = one_contour_each_line(contours)
  column_contours = one_contour_each_column(contours)

  raw_coords = split_each_line(raw_contours)
  column_coords = split_each_column(column_contours)

  cells = braille_cells(raw_coords,column_coords,img)

  parts = []
  for i in range(0,len(raw_coords)-1, 2) :
    part = img[raw_coords[i]:raw_coords[i+1] , column_coords[0]:column_coords[len(column_coords)-1]]
    parts.append(part)

  for cell in cells :
    counter = counter + 1
    if cell.shape[1] > 50:
      cell = cell[:, :45]
      spaces.append(counter)
      
    WHITE = [255,255,255]
    constant= cv2.copyMakeBorder(cell.copy(),10,10,10,0,cv2.BORDER_CONSTANT,value=WHITE)
    ret,thresh1 = cv2.threshold(constant,127,255,cv2.THRESH_BINARY)
    rgb_img = cv2.imwrite("./templates/images/cell_" + str(i) + ".png", thresh1)
    rgb_img_png = tf.keras.utils.load_img(
        "./templates/images/cell_" + str(i) + ".png",
        color_mode="rgb",
        target_size=(50,50),
        interpolation='nearest',
        keep_aspect_ratio=True
    )

    img_array = tf.keras.utils.img_to_array(rgb_img_png)
    img_array_resized = cv2.resize(img_array, (50, 50))
    img_array_final = tf.expand_dims(img_array_resized, 0) # Create a batch 

    predictions = my_model.predict(img_array_final)
    score = tf.nn.softmax(predictions[0])

    confidence = 100 * np.max(score)

    """image = cv2.imread("./templates/images/cell_" + str(i) + ".png")

    print(
          "cell_" + str(i) + " most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], confidence)
      )"""
    
    text.append(class_names[np.argmax(score)])

    i = i + 1
    for space in spaces:
      text.insert(space,"@")

    for i in range(len(text)):
        if text[i] == 'aaalef':
          text[i] = "|"
        elif text[i] == 'alef':
          text[i] = "A"
        elif text[i] == 'alef_hamza':
          text[i] = ">"
        elif text[i] == 'alef_waw':
          text[i] = "&"
        elif text[i] == 'alef_yaa':
          text[i] = "}"
        elif text[i] == 'baa':
          text[i] = "b"
        elif text[i] == 'dal':
          text[i] = "d"
        elif text[i] == 'fa':
          text[i] = "f"
        elif text[i] == 'gem':
          text[i] = "j"
        elif text[i] == 'ghain':
          text[i] = "g"
        elif text[i] == 'ha':
          text[i] = "h"
        elif text[i] == 'hha':
          text[i] = "H"
        elif text[i] == 'kaf':
          text[i] = "k"
        elif text[i] == 'kha':
          text[i] = "x"
        elif text[i] == 'lam':
          text[i] = "l"
        elif text[i] == 'meem':
          text[i] = "m"
        elif text[i] == 'noon':
          text[i] = "n"
        elif text[i] == 'qaaf':
          text[i] = "q"
        elif text[i] == 'thaa':
          text[i] = "Z"
        elif text[i] == 'thad':
          text[i] = "D"
        elif text[i] == 'tta':
          text[i] = "t"
        elif text[i] == 'ttha':
          text[i] = "v"
        elif text[i] == 'waw':
          text[i] = "w"
        elif text[i] == 'ya':
          text[i] = "y"
        elif text[i] == '@':
          text[i] = " "

    string = "".join(str(char) for char in text)
    string = pyarabic.trans.convert(string,'tim','arabic')

    for f in glob.glob("templates\images\cell_*.png"):
        os.remove(f)

  return render_template("result.html", result = string)  
        
if __name__ == "__main__":
    app.secret_key = 'braille_key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)

    app.run(debug=True)