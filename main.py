from fastapi import FastAPI, File, UploadFile
import requests
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import word_tokenize
import sklearn
from gensim.parsing.preprocessing import remove_stopwords
import os
from dotenv import load_dotenv

load_dotenv()
IMAGEDIR = "images/"



COMPUTER_VISION_KEY = os.getenv("COMPUTER_VISION_KEY")
COMPUTER_VISION_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT")

def perform_ocr(image):
    subscription_key = COMPUTER_VISION_KEY
    endpoint = COMPUTER_VISION_ENDPOINT
    ocr_url = endpoint + "vision/v3.1/ocr"

    image_data = image

    params = {'language': 'unk', 'detectOrientation': 'true'}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}

    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)

    if response.status_code == 200:
        analysis = response.json()
        return cleaned(analysis)
    else:
        return ("OCR analysis failed. Please check your credentials and try again.")


def cleaned(original_json):
    text = ""
    for line in original_json["regions"][0]["lines"]:
        for x in line["words"]:
            text = text + x["text"] + " "
    return text

def check_toxicity(text):
    model_filename = 'models/logistic_regression_model.joblib'
    loaded_model = joblib.load(model_filename)
    vectorizer_filename = 'models/tfidf_vectorizer.pkl'
    loaded_vectorizer = joblib.load(vectorizer_filename)

    
    sentence = stopword_removal(text)
    df = pd.DataFrame({'text_column': [sentence]})

    X_sentence_tfidf = loaded_vectorizer.transform(df['text_column'])

    prediction = loaded_model.predict(X_sentence_tfidf)

    prediction_dictionary = {
        "0":"Not Toxic",
        "1":"Toxic"
    }
    return prediction_dictionary[str(prediction[0])]

def stopword_removal(text):
    result = remove_stopwords(text)
    return result


app = FastAPI()
 
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)




@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    file_name = file.filename
    img = await file.read()
    result = perform_ocr(img)
    print(result)
    result = check_toxicity(result)
    return {"response":result}
