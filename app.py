import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

df = pd.read_csv('muse_v3.csv')

df['link'] = df['lastfm_url']
df['name'] = df["track"]
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name', 'emotional', 'pleasant', "link", "artist"]]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(inplace=True)
df_sad = df[:18000]
df_fear = df[18000:36808]
df_angry = df[56000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]


def fun(li):
    dat = pd.DataFrame()
    if len(li) == 1:
        v = li[0]
        t = 30
        if v == 'Neutral':
            dat = dat.append(df_neutral.sample(n=t))
        elif v == 'Angry':
            dat = dat.append(df_angry.sample(n=t))
        elif v == 'fear':
            dat = dat.append(df_fear.sample(n=t))
        elif v == "happy":
            dat = dat.append(df_happy.sample(n=t))
        else:
            dat = dat.append(df_sad.sample(n=t))
    elif len(li) == 2:
        times = [20, 10]

        for i in range(len(li)):
            v = li[i]
            t = times[i]
            if v == 'Neutral':
                dat = dat.append(df_neutral.sample(n=t))
            elif v == 'Angry':
                dat = dat.append(df_angry.sample(n=t))
            elif v == 'fear':
                dat = dat.append(df_fear.sample(n=t))
            elif v == 'happy':
                dat = dat.append(df_happy.sample(n=t))
            else:
                dat = dat.append(df_sad.sample(n=t))
    elif len(li) == 3:
        times = [15, 10, 5]

        for i in range(len(li)):
            v = li[i]
            t = times[i]
            if v == 'Neutral':
                dat = dat.append(df_neutral.sample(n=t))
            elif v == 'Angry':
                dat = dat.append(df_angry.sample(n=t))
            elif v == 'fear':
                dat = dat.append(df_fear.sample(n=t))
            elif v == 'happy':
                dat = dat.append(df_happy.sample(n=t))
            else:
                dat = dat.append(df_sad.sample(n=t))
    elif len(li) == 4:
        times = [10, 9, 8, 3]

        for i in range(len(li)):
            v = li[i]
            t = times[i]
            if v == 'Neutral':
                dat = dat.append(df_neutral.sample(n=t))
            elif v == 'Angry':
                dat = dat.append(df_angry.sample(n=t))
            elif v == 'fear':
                dat = dat.append(df_fear.sample(n=t))
            elif v == 'happy':
                dat = dat.append(df_happy.sample(n=t))
            else:
                dat = dat.append(df_sad.sample(n=t))
    else:
        times = [18, 7, 6, 5, 2]

        for i in range(len(li)):
            v = li[i]
            t = times[i]
            if v == "Neutral":
                dat = dat.append(df_neutral.sample(n=t))
            elif v == 'Angry':
                dat = dat.append(df_angry.sample(n=t))
            elif v == 'fear':
                dat = dat.append(df_fear.sample(n=t))
            elif v == "happy":
                dat = dat.append(df_happy.sample(n=t))
            else:
                dat = dat.append(df_sad.sample(n=t))
    return dat


def pre(l):
    result = [item for items, c in Counter(l).most_common() for item in [items] * c]
    temp = []
    for xt in result:
        if xt not in temp:
            temp.append(xt)
    return temp


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
st.markdown("<h2 style='text-align: center; color: white;'><b>Emotion based music recommendation</b></h2>",
            unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; color: grey;'><b>Click on the name of the recommended song to play it </b></h5>",
    unsafe_allow_html=True)
st.write("------------------------------------------------------------------------------------------------------------")
_, _, col3, _, _ = st.columns(5)
lis = []
ul = []
temp = []
with col3:
    if st.button('SCAN EMOTION'):
        count = 0
        lis.clear()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count = count + 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                lis.append(emotion_dict[maxindex])
                cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)
                lis.append(emotion_dict[maxindex])
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif count > 20:
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success('Emotion captured successfully!')

        temp = pre(lis)
        if len(temp) == 0:
            st.warning("No emotion detected. Please click on 'SCAN EMOTION' first.")
        elif len(temp) == 1:
            st.markdown(
                f"<h3 style='text-align: center; color: #f9f9f9;'>Detected Emotion: <span style='color: #f4d03f;'>{temp[0]}</span></h3>",
                unsafe_allow_html=True)
        elif len(temp) == 2:
            st.markdown(
                f"<h3 style='text-align: center; color: #f9f9f9;'>Detected Emotions: <span style='color: #f4d03f;'>{temp[0]}</span>, <span style='color: #f4d03f;'>{temp[1]}</span></h3>",
                unsafe_allow_html=True)
        elif len(temp) == 3:
            st.markdown(
                f"<h3 style='text-align: center; color: #f9f9f9;'>Detected Emotions: <span style='color: #f4d03f;'>{temp[0]}</span>, <span style='color: #f4d03f;'>{temp[1]}</span>, <span style='color: #f4d03f;'>{temp[2]}</span></h3>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<h3 style='text-align: center; color: #f9f9f9;'>Detected Emotions: <span style='color: #f4d03f;'>{temp[0]}</span>, <span style='color: #f4d03f;'>{temp[1]}</span>, <span style='color: #f4d03f;'>{temp[2]}</span>, and more...</h3>",
                unsafe_allow_html=True)
    st.text("")
    st.write("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
_, col2, _, = st.columns(3)
with col2:
    data = fun(temp)
    for ind in data.index:
        name = data['name'][ind]
        emotional = data['emotional'][ind]
        pleasant = data['pleasant'][ind]
        link = data['link'][ind]
        artist = data['artist'][ind]
        st.markdown(f"<h4 style='text-align: center; color: #f9f9f9;'><a href={link}>{name} - {artist}</a></h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #f9f9f9;'>Emotional Score: {emotional} | Pleasant Score: {pleasant}</p>", unsafe_allow_html=True)
        st.write("----------------------------------------------------------------------------------------------------")
        st.text("")