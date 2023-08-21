from flask import Flask,request,jsonify,render_template
import pickle 
import pandas as pd 
import numpy as np 
import traceback
from sklearn.preprocessing import StandardScaler
 

application = Flask(__name__)
app = application
# import linear regressor model and scaler model pickle file
standard_scaler = pickle.load(open("models/scaler.pkl","rb")) 
regressor_model = pickle.load(open("models/linear_regressor.pkl","rb"))

# route for homepage
@app.route('/')
def index():
     return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
  try:
     if request.method == 'POST':
        month = int(request.form.get('month'))
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        scaled_data = standard_scaler.transform([[month,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,Classes,Region]])
        result = regressor_model.predict(scaled_data)

        return render_template('home.html', result = result[0])
     else:
        return render_template('home.html')
  except Exception as e:
        traceback.print_exc()  # Print the exception to the console for debugging
        return "An error occurred", 500

if __name__=="__main__":
    app.run(host="0.0.0.0")
