from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Create sample data
def create_sample_data():
    data = {
        'N': np.random.randint(0, 140, 100),
        'P': np.random.randint(5, 145, 100),
        'K': np.random.randint(5, 205, 100),
        'temperature': np.random.uniform(8.83, 43.68, 100),
        'humidity': np.random.uniform(14.26, 99.98, 100),
        'ph': np.random.uniform(3.50, 9.93, 100),
        'rainfall': np.random.uniform(20.21, 298.56, 100),
        'label': np.random.choice(['rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 'apple'], 100)
    }
    return pd.DataFrame(data)

# Train model
def train_model():
    df = create_sample_data()
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']
    
    label_dict = {label: i+1 for i, label in enumerate(y.unique())}
    y = y.map(label_dict)
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mx = MinMaxScaler()
    sc = StandardScaler()
    X_train_mx = mx.fit_transform(X_train)
    X_train_sc = sc.fit_transform(X_train_mx)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sc, y_train)
    
    return model, sc, mx

app = Flask(__name__)
model, sc, mx = train_model()

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = ['Nitrogen', 'Phosporus', 'Potassium', 'Temperature', 
               'Humidity', 'pH', 'Rainfall']
    input_values = [float(request.form[f]) for f in features]
    
    single_pred = np.array(input_values).reshape(1, -1)
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)
    
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
        6: "Papaya", 7: "Orange", 8: "Apple"
    }
    
    crop = crop_dict.get(prediction[0], "Unknown")
    result = f"{crop} is the best crop to be cultivated right there"
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)