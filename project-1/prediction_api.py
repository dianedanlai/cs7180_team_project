"""
To run this app, in your terminal:
> python prediction_api.py
"""
import connexion
from sklearn.externals import joblib

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf = joblib.load('./model/predictor.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        'health'
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    prediction = clf.predict([[class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]])

    # Map the predicted value to an actual class
    if prediction[0] == 1:
        predicted_class = "Certified"
    else:
        predicted_class = "Denied"

    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Read the API definition for our service from the yaml file
app.add_api("prediction_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
