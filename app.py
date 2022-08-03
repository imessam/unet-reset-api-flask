from flask import Flask
from flask_restful import Api

from resources.prediction import Prediction

app = Flask(__name__, template_folder='templates')
api = Api(app)

api.add_resource(Prediction, "/predict")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
