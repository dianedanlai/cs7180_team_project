"""
To run this app, in your terminal:
> python prediction_api.py
"""
import connexion
from sklearn.externals import joblib
import numpy as np
import sys

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf_knn = joblib.load('models/knn.joblib')
clf_logreg = joblib.load('models/logistic_regression.joblib')
clf_rf = joblib.load('models/random_forest.joblib')
nbrs = joblib.load('models/nn10.joblib')

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
    # KNN returns a boolean result
    # Logistic Regression and Random Forest returns a probability of P("certified")
    if model == "K Nearest Neighbors":

        clf = clf_knn
        label = clf.predict([[class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]])
        return {"prediction" : "certified" if label[0] == 1 else "denied"}
    elif model == "Logistic Regression":
        clf = clf_logreg
    else:
        clf = clf_rf

    prob = clf.predict_proba([[class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]])
    return {"prediction" : prob[0][1]}

def get_similar_certified(class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089):
    # EXAMPLE OF HOW TO USE THIS MODEL.
    px = np.load("models/px.npy")
    distances, indices = nbrs.kneighbors(np.asarray([class_of_admission, country_of_citizenship, employer_city, employer_name, employer_state, pw_soc_code, pw_source_name_9089]).reshape(1, -1))
    selected = px[indices, :]
    print(selected, file=sys.stderr)
    return {"top 10 similar records": selected.tolist()}


# Read the API definition for our service from the yaml file
app.add_api("prediction_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
