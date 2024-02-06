import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import cv2
import math
import webbrowser
from gtts import gTTS
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageTk 
from playsound import playsound
from tkinter import Tk, Canvas, Button, PhotoImage, Label

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/interface")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

model = YOLO("best.pt")
classNames = ["BALL", "BLOCK", "DOG", "LYING", "MAT", "PLANK", "SITTING", "STANDING", "TREE", "WARRIOR1", "WARRIOR2"]

def start():

    canvas.delete(image_1)   
      
    label_widget = Label(window) 
    label_widget.place(width=500,height=600) 

    _, frame = vid.read() 

    frame = cv2.resize(frame, (500, 600))

    frame = cv2.flip(frame, 1)

    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cls = int(box.cls[0])

            confidence = math.ceil(box.conf[0]*100)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2

            if (classNames[cls] == "BALL") | (classNames[cls] == "BLOCK") | (classNames[cls] == "MAT"):
                color_box = (195, 153, 153)
                cv2.putText(frame, classNames[cls], [x1+3, y2-7], font, fontScale, color_box, thickness) 

            elif (classNames[cls] == "LYING") | (classNames[cls] == "SITTING") | (classNames[cls] == "STANDING"):
                color_box = (222, 140, 123)
                cv2.putText(frame, classNames[cls], [x1+3, y2-7], font, fontScale, color_box, thickness) 

            elif (classNames[cls] == "DOG") | (classNames[cls] == "PLANK") | (classNames[cls] == "TREE") | (classNames[cls] == "WARRIOR1") | (classNames[cls] == "WARRIOR2"):
                org = [x1, y1-5]
                if (confidence < 50):
                    color_box = (29, 62, 199)
                    tts('red', 'Pose detected! Hold it there...')

                elif (50 <= confidence < 75):
                    color_box = (69, 172, 244)
                    tts('yellow', 'Pose detected! Keep it up!')

                elif (75 <= confidence <= 100):
                    color_box = (0, 255, 0)
                    tts('green', 'Pose detected! Good form!')

                cv2.putText(frame, classNames[cls], (x1+4, y1+29), font, fontScale, color_box, thickness)
                cv2.putText(frame, f'{confidence}%', (x1+4, y1+57), font, fontScale, color_box, thickness)


            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 3) 

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    captured_image = Image.fromarray(opencv_image) 

    photo_image = ImageTk.PhotoImage(image=captured_image) 

    label_widget.photo_image = photo_image 

    label_widget.configure(image=photo_image) 

    label_widget.after(10, start)

    button_2['state'] = 'disabled'

    
def html():
    webbrowser.open('index.html')

def gh():
    webbrowser.open('https://github.com/adaptiveboost/asana')


def tts(name, audio):

    language = 'en-gb'

    text = gTTS(text= f'{audio}',
                lang= language,
                slow= False)
        
    text.save(f'{name}.mp3')
    playsound(f'{name}.mp3', block=False)
    os.remove(f'{name}.mp3')

vid = cv2.VideoCapture(0) 

window = Tk()
window.iconbitmap('./assets/icon.ico')

tts('welcome', "Welcome to Asana! Click what's this for more information, or click me to begin.")
playsound('./assets/music.wav', block=False)

window.title("ASANA - Ascend with artificial intelligence")
window.geometry("900x600")
window.configure(bg = "#F3F6F6")

canvas = Canvas(
    window,
    bg = "#F3F6F6",
    height = 600,
    width = 900,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    250.0,
    300.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    704.0,
    70.0,
    image=image_image_2
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: html(),
    relief="flat"
)
button_1.place(
    x=769.0,
    y=360.0,
    width=85.0,
    height=28.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: start(),
    relief="flat"
)
button_2.place(
    x=555.0,
    y=279.0,
    width=300.0,
    height=75.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: gh(),
    relief="flat"
)
button_3.place(
    x=850.0,
    y=559.0,
    width=30.0,
    height=30.0
)
window.resizable(False, False)

window.mainloop()
