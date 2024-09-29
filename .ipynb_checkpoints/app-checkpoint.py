from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your pre-trained pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('dataframe.pkl', 'rb') as f:
    df = pickle.load(f)

@app.route('/',methods=['GET','POST'])
def home():
    locations=df['Location'].unique().tolist()
    return render_template('index1.html',locations=locations)
@app.route('/result', methods=['GET', 'POST'])
def result():
    prediction = None
    if request.method == 'POST':
        # Fetch form data
        area = int(request.form['area'])
        location = request.form['location']
        bhk = int(request.form['bhk'])

        # Handle checkboxes correctly
        Resale = 1 if 'Resale' in request.form else 0
        Furnished = 1 if 'Furnished' in request.form else 0
        parking = 1 if 'parking' in request.form else 0
        
        gardens = 1 if 'gardens' in request.form else 0


        
        # Prepare the data for prediction
        features = np.array([[gardens, Resale, area, parking, Furnished,location,
                            bhk]])

        a_df=pd.DataFrame(features,columns=['Garden', 'Resale', 'Area', 'Parking', 'Furnished', 'Location', 'BHK'])

        # Prediction using the pipeline
        prediction = pipeline.predict(a_df)[0]
        prediction=round(prediction,3)
        if prediction >= 1:
            formatted_prediction = f"{prediction:.2f} Cr"
        else:
            formatted_prediction = f"{prediction * 100:.2f} Lakhs"

    # Render the result template with the prediction
    return render_template('result.html', prediction=formatted_prediction)

    

# @app.route('/result')
# def result():
#     # Get the prediction from the query parameters
#     prediction = request.args.get('prediction', type=float)
#     return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
