"""Generate demonstration data and train an ANN. Synthetic data is not evidence of real-world accuracy."""
import os, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FEATURES = ['study_hours','attendance','previous_marks','assignment_score','internal_marks','test_scores','sleep_hours','internet_access','class_participation','historical_performance']

def make_data(n=1600, seed=42):
    rng=np.random.default_rng(seed)
    study=rng.uniform(0,8,n); attendance=rng.uniform(45,100,n); prev=rng.uniform(30,95,n)
    assignment=rng.uniform(20,100,n); internal=rng.uniform(20,100,n); tests=rng.uniform(20,100,n)
    sleep=rng.uniform(3,10,n); internet=rng.integers(0,2,n); participation=rng.uniform(0,10,n); history=rng.uniform(30,95,n)
    X=np.column_stack([study,attendance,prev,assignment,internal,tests,sleep,internet,participation,history])
    y=np.clip(.12*study*5 + .18*attendance + .16*prev + .12*assignment + .16*internal + .12*tests + .04*sleep*8 + .04*internet*100 + .04*participation*10 + .02*history + rng.normal(0,5,n),0,100)
    return X,y

if __name__=='__main__':
    X,y=make_data(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
    model=Sequential([Dense(64,activation='relu',input_shape=(10,)),Dropout(.2),Dense(32,activation='relu'),Dense(16,activation='relu'),Dense(1)])
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    model.fit(Xtr,ytr,epochs=120,batch_size=32,validation_split=.15,callbacks=[EarlyStopping(patience=12,restore_best_weights=True)],verbose=1)
    pred=model.predict(Xte,verbose=0).ravel()
    os.makedirs('model',exist_ok=True); model.save('model/ann.keras'); joblib.dump(scaler,'model/scaler.joblib')
    metrics={'mae':float(mean_absolute_error(yte,pred)),'r2':float(r2_score(yte,pred)),'data':'synthetic demonstration'}
    with open('model/metrics.json','w') as f: json.dump(metrics,f,indent=2)
    print(metrics)
