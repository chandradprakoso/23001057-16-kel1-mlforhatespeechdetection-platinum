import re
import pandas as pd
import sqlite3


from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)


swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing and Modeling'),
    },
    host= LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    'static_url_path': "/flasgger_static",
    'swagger_ui': True,
    'specs_route': "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                  config=swagger_config)



model_nn = pickle.load(open(".venv/modelplatinumneuralnetwork.p", "rb"))
vect_nn = pickle.load(open(".venv/featureplatinumneuralnetwork.p", "rb"))
 

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
sentiment = ['negative', 'neutral', 'positive']

file = open(".venv/x_pad_sequencesplatinumLSTM.pickle", 'rb')
feature_lstm = pickle.load(file)
file.close()

model_lstm = load_model('.venv/modelplatinumLSTM.h5')

def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    text = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text



@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "Halaman API Platinum Challenge Kelompok 1 BINAR DSC Wave 16",
        'data': "Untuk masuk ke halaman Dokumentasi, silakan masuk klik link berikut 127.0.0.1:5000/docs/ ",
    }

    response_data = jsonify(json_response)
    return response_data


#NeuralNetwork
@swag_from("docs/text_processing_nn.yml", methods = ['POST'])
@app.route('/text-processing_nn', methods=['POST'])

def nn_text_processing():

    text = request.form.get('text')

    text_final = vect_nn.transform([cleansing(text)])

    result = model_nn.predict(text_final)
    resultjson = result.tolist()[0]

    json_response = {
        'status_code': 200,
        'description': "Teks yang skudah diproses",
        'data_raw': text, 
        'data_clean': resultjson
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/filepostmethod_nn.yml", methods = ['POST'])
@app.route('/file-processing_nn', methods=['POST'])

def nn_file_processing():
        
    file = request.files.getlist('file')[0]

    df = pd.read_csv(file, encoding='ISO-8859-1')
    texts = df.Tweet.to_list()

    text_clean = []
    for text in texts:
        text_clean.append(cleansing(text))

    text_vect = vect_nn.transform(text_clean)

    result = []
    for text in text_vect:
        result.append(model_nn.predict(text))
    
    array = np.array(result)
    
    resultjson =  array.tolist()

    json_response = {
        'data_row' : texts,
        'data_clean' : resultjson
    }

    response_data = jsonify(json_response)
    return response_data

#LSTM
@swag_from("docs/text_processing_lstm.yml", methods = ['POST'])
@app.route('/text-processing_lstm', methods=['POST'])

def lstm_text_processing():

    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_lstm.shape[1])

    prediction = model_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    
    json_response = {
        'status_code': 200,
        'description': "Teks yang skudah diproses",
        'data_raw': original_text, 
        'data_clean': get_sentiment
    }
    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/filepostmethod_lstm.yml", methods = ['POST'])
@app.route('/file-processing_lstm', methods=['POST'])

def lstm_file_processing():
        
    file = request.files.getlist('file')[0]

    df = pd.read_csv(file, encoding='ISO-8859-1')
    texts = df.Tweet.to_list()

    text_clean = []
    for text in texts:
        text_clean.append(cleansing(text))
    
    result = []
    for x in text_clean:
        feature = tokenizer.texts_to_sequences(x)
        feature = pad_sequences(feature, maxlen=feature_lstm.shape[1])
        prediction = model_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        result.append(get_sentiment)
        
    array = np.array(result)
    
    resultjson =  array.tolist()

    json_response = {
        'data_row' : texts,
        'data_clean' : resultjson
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True)
