from flask import Flask, render_template, request
from Neural import NeuralNetwork
import cv2
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smit'
app.config['UPLOAD_FOLDER'] = 'static'

# Debug setting set to true
app.debug = True
NN = NeuralNetwork()


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.get('/')
def index():
    return render_template("index.html")


@app.post('/application')
def predict_cavity():
    file = request.files['file']  # get the uploaded file
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    pre = NN.prediction(img)


    # form = UploadFileForm()
    # if form.validate_on_submit():
    #     file = form.file.data  # First grab the file
    #     print(file)
    #     file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
    #                            secure_filename("image.jpg")))

    # pre = NN.prediction(request.data)
    return render_template("result.html", data=pre)


@app.get('/application')
def home():
    return render_template("home.html", form=UploadFileForm())


@app.get('/modelInfo')
def model_info():
    args = request.args.to_dict()
    return NN.model_info(args.get("number"))


if __name__ == '__main__':
    app.run()
