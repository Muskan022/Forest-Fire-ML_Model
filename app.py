from flask import Flask,request,jsonify,render_template
import pickle 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
 

app = Flask(__name__)

# import linear regressor model and scaler model pickle file
standard_scaler = pickle.load(open(".vscode/models/scaler.pkl","rb")) 
regressor = pickle.load(open(".vscode/models/linear_regressor.pkl","rb"))

# route for homepage
@app.route('/')
def index():
    render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        month = float(request.form.get('month'))
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        scaled_data = standard_scaler.transform([[month,Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = regressor.predict(scaled_data)

        return render_template('home.html',result = result[0])
    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")
