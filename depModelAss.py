from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import preProPortugal
import numpy as np


app = Flask(__name__)

#we identify and load the model in MLFlow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_name="assignmentFTry"
model_version="21"
model_uri = f"models:/{model_name}/{model_version}"  
model = mlflow.pyfunc.load_model(model_uri)



@app.route('/predict', methods=['POST'])
def predict():

    #we get the list with the inputs
    data = request.get_json()

    if 'features' not in data:
        return jsonify({'error': "La clave 'features' no esta presente en la solicitud"}), 400  

    elif not isinstance(data['features'], list):
        return jsonify({'error': 'Invalid input: "features" must be a list of lists'}), 400
    
    input_data = data['features']

    if not all(isinstance(row, list) and len(row) == 12 for row in input_data):
        return jsonify({'error': 'Each row in "features" must be a list of 12 numbers'}), 400
    
    #we preprocess the data before feeding it to the algorithm

    df_input=preProPortugal.encodeData(input_data)

    df_input_scaled=preProPortugal.preprocess_data(df_input)

    #we predict the output and scale it, then we return it

    prediction=model.predict(df_input_scaled)

    listp= prediction.tolist()
    
    pred_sc = [[np.exp(value) for value in inner_list] for inner_list in listp]

    return jsonify({'prediction':pred_sc})

    
   

"""if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)"""



if __name__ == '__main__':
    app.run(debug=True)


