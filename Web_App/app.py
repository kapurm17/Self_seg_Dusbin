from flask import Flask, render_template, request
from predict import recreate_model, get_tensor, predict_img
from inference import get_class
import io
from PIL import Image

model =recreate_model('./weights.pth')
app = Flask(__name__)

@app.route('/', methods= ['GET', 'POST'])

def func_one():
    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File not uploaded')
            return

        file = request.files['file']
        img = file.read()
        f = io.BytesIO(img)
        predictions= predict_img(model, f)
        return render_template('predictor.html', prediction=predictions)



if __name__=='__main__':
    app.run(debug=True)
