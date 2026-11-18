import os, sqlite3, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, pandas as pd

app=Flask(__name__); CORS(app)
DB='edupredict.db'; MODEL_PATH='model/ann.keras'; SCALER_PATH='model/scaler.joblib'
FEATURES=['study_hours','attendance','previous_marks','assignment_score','internal_marks','test_scores','sleep_hours','internet_access','class_participation','historical_performance']

def db():
 c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
 with db() as c:
  c.execute('CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE)')
  c.execute('CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, student_id INTEGER, data TEXT, predicted REAL, category TEXT, risk TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')
init_db()

def load_model():
 if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
  from tensorflow.keras.models import load_model
  import joblib
  return load_model(MODEL_PATH),joblib.load(SCALER_PATH)
 return None,None

def category(m): return 'Excellent' if m>=85 else 'Good' if m>=70 else 'Average' if m>=50 else 'Needs Improvement'
def risk(d,p):
 score=sum([d['attendance']<70,d['study_hours']<1.5,d['internal_marks']<45,d['assignment_score']<45,d['test_scores']<45,d['historical_performance']<50])
 return 'High Risk' if score>=4 or p<45 else 'Medium Risk' if score>=2 or p<60 else 'Low Risk'

def predict(d):
 model,scaler=load_model()
 if model is None: return None,'ANN not trained yet. Run python train.py in backend.'
 x=np.array([[float(d[k]) for k in FEATURES]]); p=float(np.clip(model.predict(scaler.transform(x),verbose=0)[0][0],0,100)); return p,None

@app.get('/api/health')
def health(): return jsonify(status='ok',model_ready=os.path.exists(MODEL_PATH))

@app.post('/api/predict')
def prediction():
 d=request.get_json(force=True)
 try:
  for k in FEATURES:
   if k not in d: return jsonify(error=f'Missing {k}'),400
  for k in FEATURES:
   d[k]=float(d[k])
   if not np.isfinite(d[k]): return jsonify(error=f'Invalid {k}'),400
  if not (0<=d['attendance']<=100 and 0<=d['assignment_score']<=100 and 0<=d['internal_marks']<=100 and 0<=d['test_scores']<=100 and 0<=d['previous_marks']<=100 and 0<=d['historical_performance']<=100): return jsonify(error='Marks and attendance must be between 0 and 100'),400
  p,e=predict(d)
  if e: return jsonify(error=e),503
  cat=category(p); r=risk(d,p)
  reasons=[]
  for k,label,threshold in [('attendance','Attendance',70),('study_hours','Study time',1.5),('internal_marks','Internal marks',45),('assignment_score','Assignment score',45),('test_scores','Test scores',45)]:
   if d[k]<threshold: reasons.append(f'{label} is below the support threshold')
  # Transparent input-based influence summary; model-specific SHAP explanation is a future refinement.
  importance={k:float(v) for k,v in zip(FEATURES,model_input_weights(d))}
  with db() as c: c.execute('INSERT INTO records(data,predicted,category,risk) VALUES(?,?,?,?)',(json.dumps(d),p,cat,r))
  return jsonify(predicted_marks=round(p,2),category=cat,risk=r,reasons=reasons,feature_summary=importance)
 except Exception as ex: return jsonify(error=str(ex)),400

def model_input_weights(d):
 # Heuristic display of feature values, explicitly not SHAP/model attribution.
 vals=[d[k] for k in FEATURES]; arr=np.asarray(vals,dtype=float); denom=np.sum(np.abs(arr))+1e-8
 return np.abs(arr)/denom

@app.post('/api/simulate')
def simulate():
 d=request.get_json(force=True); changes=d.pop('changes',{})
 try:
  base,e=predict(d)
  if e:return jsonify(error=e),503
  d.update(changes); after,e=predict(d)
  if e:return jsonify(error=e),503
  return jsonify(original=round(base,2),simulated=round(after,2),difference=round(after-base,2))
 except Exception as ex:return jsonify(error=str(ex)),400

@app.post('/api/subjects')
def subjects():
 d=request.get_json(force=True); marks=d.get('subject_marks',{})
 if not marks:return jsonify(error='Provide subject_marks object'),400
 vals={k:float(v) for k,v in marks.items()}; avg=sum(vals.values())/len(vals)
 weak=[k for k,v in vals.items() if v<60]; strong=[k for k,v in vals.items() if v>=80]
 return jsonify(subjects=vals,average=round(avg,2),weak=weak,strong=strong,suggestions=[f'Practice and seek faculty guidance in {s}' for s in weak])

@app.get('/api/history')
def history():
 with db() as c: rows=c.execute('SELECT id,data,predicted,category,risk,created_at FROM records ORDER BY id DESC LIMIT 100').fetchall()
 return jsonify([dict(r) for r in rows])

@app.post('/api/upload')
def upload():
 f=request.files.get('file')
 if not f:return jsonify(error='CSV file required'),400
 try:
  df=pd.read_csv(f); return jsonify(rows=len(df),columns=list(df.columns),message='CSV parsed. Map columns to model inputs before batch prediction.' )
 except Exception as ex:return jsonify(error=str(ex)),400

if __name__=='__main__': app.run(host='0.0.0.0',port=5000,debug=True)
