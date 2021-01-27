from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Instantiate a flask object, serve from this file
app = Flask(__name__)

# Load models
flag_model = pickle.load(open('sgd_binary.pickle','rb'))
#rank_model = pickle.load(open('./models/------.pickle','rb'))

# Map the route (end of URL) to each function
@app.route('/')
def index():
    # Render homepage
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # Get review from user
        review = request.form.get("in_review")

        # Predict
        flagged = flag_model.predict([review])
        #rank = rank_model.predict([review])

        rank = flagged

        if flagged == 'flag':
            return render_template("flag_result.html", pred = rank)
        else:
            return render_template("pass_result.html", pred = rank)

    else:
        return render_template("index.html")


@app.route('/about')
def about():
    # Render homepage
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True) 
