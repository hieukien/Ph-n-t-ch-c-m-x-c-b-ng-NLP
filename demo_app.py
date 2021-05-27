from flask import Flask, render_template, url_for, request, jsonify
import os
import pickle
import tensorflow
import tensorflow
import pickle
import pickle as pkl





with open("train.pkl", "rb") as f:
    train_x, train_y = pkl.load(f)
with open("test.pkl", "rb") as f:
    test_x, test_y = pkl.load(f)
with open("traint.pkl", "rb") as f:
    traint_x1, traint_y2 = pkl.load(f)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer(num_words=10000,oov_token="<OOV>")
tokenizer.fit_on_texts(traint_x1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Embedding
from tensorflow.keras.layers import  Flatten
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Embedding(10000,64,input_length=140))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()
model.fit(train_x,train_y,epochs=1,validation_data=(test_x,test_y))

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route("/")
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'abc.jpg')
    return render_template('index.html', user_image = pic1)

@app.route("/predict", methods=['POST'])
def predict():
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]
    test=tokenizer.texts_to_sequences(data)
    padd_test=pad_sequences(test,maxlen=140,truncating='post',padding='post')
    pred=model.predict(padd_test)
    a=pred
    if a >0.5:
        b="Tích cực"
    if a <0.5:
        b="Tiêu cực"
   
    return render_template('index.html', 
                            b = 'Đây là comment: {}'.format(b)
                           )
                            


    


#if __name__ == '__main__':
app.run(debug=True)
