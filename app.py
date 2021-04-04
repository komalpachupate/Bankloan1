from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'ubl.pkl'
classifier= pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            #  reading the inputs given by the user
            Age=float(request.form['Age'])
            Experience = float(request.form['Experience'])
            Income = float(request.form['Income'])
            Family = float(request.form['Family'])
            CCAvg = float(request.form['CCAvg'])
            Education = float(request.form['Education'])
            Mortgage = float(request.form['Mortgage'])
            Securities_Account = float(request.form['Securities Account'])
            CD_Account = float(request.form['CD Account'])
            Online = float(request.form['Online'])
            CreditCard = float(request.form['CreditCard'])
            
            
            data=np.array([[Age,Experience,Income,Family,CCAvg,Education,Mortgage,Securities_Account,CD_Account,Online,CreditCard]])
            my_prediction = classifier.predict(data)
        
            return render_template('results.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)