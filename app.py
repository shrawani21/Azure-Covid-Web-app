import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    df = pd.DataFrame(final_features, columns = ['cough','fever','sore_throat','shortness_of_breath','head_ache','age_60_and_above','test_indication_Abroad','test_indication_Contact_with_Confirmed','test_indication_Others'])

    prediction = model.predict(df)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Chances that you are Covid Positive: {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
