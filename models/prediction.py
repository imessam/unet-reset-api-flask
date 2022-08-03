from utils import run
from flask import render_template,make_response
import urllib.request


class PredictionModel:

    @classmethod
    def predict(cls, url):
        local_filename, headers = urllib.request.urlretrieve(url)

        filename = run.run(local_filename, "weights/unet.weights")
        return make_response(render_template('view.html', image_path=filename))
