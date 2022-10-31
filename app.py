from flask import Flask, render_template, request
from weather_app.classifier import Classifier
import logging
import os

app =Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def prt():
    return render_template('index.html')

@app.route('/temp', methods= ["GET", "POST"])
def temp():
    try:
        outlook = request.form['weather']
        temp = request.form['temperature']
        humidity = request.form['Humidity']
        windy = request.form['Windy']
        logging.info('Input taken from the user')
        path = os.path.join(os.getcwd(), 'Data/dataset.csv')

        c = Classifier(filename=path, class_attr="Play")
        logging.info('Instance of Classifier created')

        c.hypothesis = {"Outlook":outlook, "Temp":temp, "Humidity":humidity , "Windy":windy}
        prior = c.calculate_priori()
        logging.info('Prior probabilities calculated')

        cond =  c.calculate_conditional_probabilities(c.hypothesis)
        logging.info('Conditional probabilities calculated')

        posterior =  c.classify()
        logging.info('Posterior probabilities calculated')

        return render_template('results.html',pri_prob=prior, dict1=cond['yes'],dict2=cond['no'], post_prob = posterior)
    except Exception as e:
        return e

if __name__ =='__main__':
    # app.run(host='0.0.0.0', port=5000)
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)