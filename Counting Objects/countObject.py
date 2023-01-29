import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template

image = cv2.imread('images/soap2_w.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blur, 30, 150, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)

(cnt, hierarchy) = cv2.findContours(
	dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)


print("Objects in the image : ", len(cnt))
ranvar = str(len(cnt))

app = Flask(__name__)
@app.route('/')
def index():
    return ranvar

if __name__ == "__main__":
    app.run(debug=True)