from flask_restful import Resource, reqparse
from models import prediction


class Prediction(Resource):

    parser = reqparse.RequestParser()

    parser.add_argument(
        "url",
        type=str,
        required=True,
        help="This field is necessary"
    )

    def get(self):
        url = self.parser.parse_args()["url"]

        return prediction.PredictionModel.predict(url)
