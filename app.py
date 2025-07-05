from flask import Flask, request, Response
import pickle
import pandas as pd
import os
from codes.run_model import PredictCancel

# Carregar o modelo
model = pickle.load(open('model/final_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/codes', methods=['POST'])
def cancel_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        pipeline = PredictCancel()
        df_cleaning = pipeline.data_cleaning(test_raw)
        df_feature = pipeline.feature_engineering(df_cleaning)
        df_preparation = pipeline.data_preparation(df_feature)
        df_predict = pipeline.get_predictions(model, df_preparation, test_raw)

        return df_predict
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
