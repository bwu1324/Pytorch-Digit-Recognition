import numpy as np
import torch
from flask import Flask
from flask import request, send_file

from dataset import crop_in

# Index HTML File Location
INDEX_PATH = "index.html"

# Load TorchScript Model
MODEL_PATH = "models/2_conv_2_linear.pt"
model = torch.jit.load(MODEL_PATH)
model.eval()

app = Flask(__name__)


@app.route("/")
def index():
    return send_file(INDEX_PATH)


@app.route("/recognize_digit", methods=["POST"])
def recognize_digit():
    # Get pixel array
    pixelarray = request.args.get('pixelarray').split(',')

    # Run prediction
    X = crop_in(np.array(pixelarray, dtype='float32') * 255)
    y = model(X.reshape((1, 1, 28, 28)))

    # Convert one-hot output to number
    numpy_predict = y.detach().numpy().flatten()
    return str(np.argmax(numpy_predict))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
