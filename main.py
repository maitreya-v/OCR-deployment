from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from paddleocr import PaddleOCR
import regex as re
from PIL import Image
import os
import cv2
import io
import requests
from io import BytesIO
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import word_tokenize
import sklearn
IMAGEDIR = "images/"

class ChatOcr:
    def __init__(self, lang = "en"):
        self.model = PaddleOCR(use_angle_cls=True, lang=lang)
    
    def ocr(self, img_bytes):
        img = Image.open(BytesIO(img_bytes))
        width, height = img.size

        result = self.model.ocr(img_bytes, cls=True)[0]

        filtered_result = self._filter_ocr_ouputs(result)

        classified_result = self._classify(filtered_result, width, height)

        return classified_result

    def _is_date(self, text):
        text = text.replace(" ", "")

        pattern = r"^([0-9]{1,2}\-[0-9]{1,2}\-[0-9]{2})?.{,2}[0-9]{1,2}\:[0-9]{1,2}[AP]M$"
        match = re.search(pattern, text)
        
        if match:
            return True
        else:
            return False

    def _classify(self, filtered_result, width, height):


        classified_result = []

        for line in filtered_result:
            center_x = (line["bbox"][0][0] + line["bbox"][1][0]) / 2
            mean_y = (line["bbox"][0][1] + line["bbox"][1][1]) / 2

            if center_x / width < 0.5:
                line["sent_by"] = "A"
            else:
                line["sent_by"] = "B"

            line["mean_y"] = mean_y
            classified_result.append(line)
        
        mixed_classified_result = []
        discarded_index = []

        for i in range(len(classified_result) - 1):
            if i in discarded_index:
                continue

            if abs(classified_result[i]["mean_y"] - classified_result[i + 1]["mean_y"])/height < 0.05:
                mixed_classified_result.append({
                    "text": classified_result[i]["text"] + classified_result[i + 1]["text"],
                    "sent_by": classified_result[i]["sent_by"],
                })
                discarded_index.append(i + 1)
            
            else:
                mixed_classified_result.append({
                    "text": classified_result[i]["text"],
                    "sent_by": classified_result[i]["sent_by"],
                })    
                
        
        if len(classified_result) - 1 not in discarded_index:
            mixed_classified_result.append({
                "text": classified_result[-1]["text"],
                "sent_by": classified_result[-1]["sent_by"],
            })
                
        
        return mixed_classified_result


    def _filter_ocr_ouputs(self, ocr_outputs):

        result = []

        for line in ocr_outputs:
            text = line[-1][0]
            conf = line[-1][1]

            if self._is_date(text):
                continue

            result.append({
                "text": text,
                "conf": conf,
                "bbox": line[0]
            })

        return result


def check_toxicity(text):
    model_filename = 'models/logistic_regression_model.joblib'
    loaded_model = joblib.load(model_filename)
    nltk.download("stopwords")
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
    stop_words = set(stopwords.words('english'))
  
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)


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
    ocr = ChatOcr()
    img = await file.read()
    result = ocr.ocr(img)
    distribution = {
        "A":[],
        "B":[]
    }
    for line in result:
        if line["sent_by"]=="A":
            distribution["A"].append(line["text"])
        else:
            distribution["B"].append(line["text"])
    result = check_toxicity(" ".join(distribution["A"]))
    return {"response":result}

