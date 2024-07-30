# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the pipeline
# with open('pipeline.pkl', 'rb') as f:
#     pipe = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     return 

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # Fetch form data
#         area = request.form.get('area')
#         location = request.form.get('location')
#         bhk = request.form.get('bhk')
#         new_resale = request.form.get('newResale')
#         lift_available = 'liftAvailable' in request.form
#         car_parking = 'carParking' in request.form
#         gas_connection = 'gasConnection' in request.form
#         area_per_room = request.form.get('areaPerRoom')
#         price_per_sqft = request.form.get('pricePerSqft')

#         # Print or process the data as needed
#         print(f'Area: {area}')
#         print(f'Location: {location}')
#         print(f'BHK: {bhk}')
#         print(f'New/Resale: {new_resale}')
#         print(f'Lift Available: {lift_available}')
#         print(f'Car Parking: {car_parking}')
#         print(f'Gas Connection: {gas_connection}')
#         print(f'Area per Room: {area_per_room}')
#         print(f'Price per Sq ft: {price_per_sqft}')

#         # Example response
#         a= f"Form submitted successfully! Area: {area}, Location: {location}"

#     return render_template('index.html')

# @app.route('/predict')
# def prd():
#     return "thskfjs"
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained pipeline
with open('pipeline.pkl', 'rb') as f:
     pipeline = pickle.load(f)
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Fetch form data
        area = float(request.form['area'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        new_resale = 1 if request.form['newResale'] == 'new' else 0
        lift_available = 1 if 'liftAvailable' in request.form else 0
        car_parking = 1 if 'carParking' in request.form else 0
        gas_connection = 1 if 'gasConnection' in request.form else 0
        area_per_room = float(request.form['areaPerRoom'])
        price_per_sqft = float(request.form['pricePerSqft'])

        # Prepare the feature array
        features = np.array([[area, location, bhk, new_resale, lift_available,
                              car_parking, gas_connection, area_per_room, price_per_sqft]])

        # Predict using the pipeline
        prediction = pipeline.predict(features)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
