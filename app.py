from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React Native app

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')  # Ensure this is the correct model
scaler = joblib.load('scaler.pkl')  # Ensure this is the correct scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/location')
def location():
    return render_template('location.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form

        # Extract features
        features = [
            int(data['bedrooms']), int(data['grade']), int(data['has_basement']), float(data['living_in_m2']),
            int(data['renovated']), int(data['nice_view']), int(data['perfect_condition']), int(data['real_bathrooms']),
            int(data['has_lavatory']), int(data['single_floor']), int(data['month']), int(data['quartile_zone']), int(data['year'])
        ]

        # Prepare input data
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]

        # Ensure predicted price is non-negative
        predicted_price = max(predicted_price, 0)  # Prevents negative values

        # Return JSON response for mobile app
        if request.content_type == 'application/json':
            return jsonify({'prediction': round(predicted_price, 2)})
        
        # Render result for web app
        return render_template('result.html', price=round(predicted_price, 2))
    
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
