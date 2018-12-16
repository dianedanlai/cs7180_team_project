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
clf_knn = joblib.load('models/knn.joblib')
clf_logreg = joblib.load('models/logistic_regression.joblib')
clf_rf = joblib.load('models/random_forest.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        'health'
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089, model):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    if model == "K Nearest Neighbors":
        clf = clf_knn
    elif model == "Logistic Regression":
        clf = clf_logreg
    else:
        clf = clf_rf
    label = clf.predict([[class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]])
    prob = clf.predict_proba([[class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]])

    # Map the predicted value to an actual class
    if label[0] == 0:
        prob = 1 - prob

    # Return the prediction as a json
    return {"prediction" : "probability of being certified is " + prob}

def get_similar_certified(class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089, model):
    if model == "K Nearest Neighbors":
        clf = clf_knn
    elif model == "Logistic Regression":
        clf = clf_logreg
    else:
        clf = clf_rf

    results = ""

    return


# Read the API definition for our service from the yaml file
app.add_api("prediction_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
