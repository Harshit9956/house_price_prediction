from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index1.html')
@app.route('/result', methods=['GET', 'POST'])
def result():
    prediction = None
    if request.method == 'POST':
        # Fetch form data
        area = int(request.form['area'])
        location = request.form['location']
        bhk = int(request.form['bhk'])

        # Handle checkboxes correctly
        new = 1 if 'new' in request.form else 0
        gymnasium = 1 if 'gymnasium' in request.form else 0
        lift_available = 1 if 'lift_available' in request.form else 0
        car_parking = 1 if 'car_parking' in request.form else 0
        childrens_play_area = 1 if 'childrens_play_area' in request.form else 0
        landscaped_gardens = 1 if 'landscaped_gardens' in request.form else 0
        indoor_games = 1 if 'indoor_games' in request.form else 0
        gas_connection = 1 if 'gas_connection' in request.form else 0
        jogging_track = 1 if 'jogging_track' in request.form else 0
        swimming_pool = 1 if 'swimming_pool' in request.form else 0

        garden_area=landscaped_gardens+childrens_play_area+jogging_track
        # Prepare the data for prediction
        features = np.array([[area, location, bhk, new, gymnasium, lift_available, car_parking,
                             indoor_games, gas_connection,
                               swimming_pool,garden_area]])

# ['Area', 'Location', 'Bhk', 'New', 'Gymnasium', 'Lift Available',
#        'Car Parking', 'Indoor Games', 'Gas Connection', 'Swimming Pool',
#        'Garden_area'

        # Prediction using the pipeline
        prediction = pipeline.predict(features)[0]
        prediction=round(prediction,2)

    return render_template('result.html',prediction=prediction)

# @app.route('/result')
# def result():
#     # Get the prediction from the query parameters
#     prediction = request.args.get('prediction', type=float)
#     return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
