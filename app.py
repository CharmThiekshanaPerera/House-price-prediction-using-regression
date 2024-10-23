from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')  # Make sure this is your saved model
scaler = joblib.load('scaler.pkl')  # Make sure this is your saved scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        bedrooms = int(request.form['bedrooms'])
        grade = int(request.form['grade'])
        has_basement = int(request.form['has_basement'])
        living_in_m2 = float(request.form['living_in_m2'])
        renovated = int(request.form['renovated'])
        nice_view = int(request.form['nice_view'])
        perfect_condition = int(request.form['perfect_condition'])
        real_bathrooms = int(request.form['real_bathrooms'])
        has_lavatory = int(request.form['has_lavatory'])
        single_floor = int(request.form['single_floor'])
        month = int(request.form['month'])
        quartile_zone = int(request.form['quartile_zone'])
        year = int(request.form['year'])
        
        # Prepare input data in the correct order
        input_data = np.array([[bedrooms, grade, has_basement, living_in_m2, renovated, nice_view, 
                                perfect_condition, real_bathrooms, has_lavatory, single_floor, 
                                month, quartile_zone, year]])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make the prediction
        predicted_price = model.predict(input_scaled)
        
        # Display result
        return render_template('result.html', price=round(predicted_price[0], 2))

if __name__ == '__main__':
    app.run(debug=True)



