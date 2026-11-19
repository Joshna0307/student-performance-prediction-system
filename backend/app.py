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
# Load compact ANN weights once. TensorFlow is used for training, not serving,
# keeping prediction workers small enough for low-memory hosting.
_INFERENCE_CACHE=None
def load_model():
 global _INFERENCE_CACHE
 if _INFERENCE_CACHE is not None:return _INFERENCE_CACHE
 weights_path='model/ann_weights.npz'
 if os.path.exists(weights_path) and os.path.exists(SCALER_PATH):
  import joblib
  archive=np.load(weights_path)
  weights=[archive[f'arr_{i}'] for i in range(8)]
  _INFERENCE_CACHE=(weights,joblib.load(SCALER_PATH))
  return _INFERENCE_CACHE
 return None,None
def category(m): return 'Excellent' if m>=85 else 'Good' if m>=70 else 'Average' if m>=50 else 'Needs Improvement'
def risk(d,p):
 score=sum([d['attendance']<70,d['study_hours']<1.5,d['internal_marks']<45,d['assignment_score']<45,d['test_scores']<45,d['historical_performance']<50])
 return 'High Risk' if score>=4 or p<45 else 'Medium Risk' if score>=2 or p<60 else 'Low Risk'
def predict(d):
 weights,scaler=load_model()
 if weights is None:return None,'ANN not trained yet. Run python train.py in backend.'
 x=scaler.transform(np.array([[float(d[k]) for k in FEATURES]],dtype=float))
 # Dense(ReLU) -> Dropout(no-op at inference) -> Dense(ReLU) -> Dense(ReLU) -> Dense(linear)
 for i in range(0,8,2):
  x=x @ weights[i] + weights[i+1]
  if i<6:x=np.maximum(x,0)
 return float(np.clip(x[0,0],0,100)),None
@app.get('/api/health')
def health():return jsonify(status='ok',model_ready=os.path.exists(MODEL_PATH))
@app.post('/api/predict')
def prediction():
 d=request.get_json(force=True)
 try:
  for k in FEATURES:
   if k not in d:return jsonify(error=f'Missing {k}'),400
   d[k]=float(d[k])
   if not np.isfinite(d[k]):return jsonify(error=f'Invalid {k}'),400
  for k in ['attendance','previous_marks','assignment_score','internal_marks','test_scores','historical_performance']:
   if not 0<=d[k]<=100:return jsonify(error=f'{k} must be between 0 and 100'),400
  if not 0<=d['study_hours']<=24 or not 0<=d['sleep_hours']<=24 or not 0<=d['class_participation']<=10 or d['internet_access'] not in [0,1]:return jsonify(error='Study/sleep hours, participation or internet value is invalid'),400
  p,e=predict(d)
  if e:return jsonify(error=e),503
  cat=category(p); r=risk(d,p)
  reasons=[]
  for k,label,threshold in [('attendance','Attendance',70),('study_hours','Study time',1.5),('internal_marks','Internal marks',45),('assignment_score','Assignment score',45),('test_scores','Test scores',45),('historical_performance','Historical performance',50)]:
   if d[k]<threshold:reasons.append(f'{label} is below the support threshold')
  with db() as c:c.execute('INSERT INTO records(data,predicted,category,risk) VALUES(?,?,?,?)',(json.dumps(d),p,cat,r))
  return jsonify(predicted_marks=round(p,2),category=cat,risk=r,reasons=reasons)
 except Exception as ex:return jsonify(error=str(ex)),400
@app.post('/api/simulate')
def simulate():
 payload=request.get_json(force=True); changes=payload.get('changes',{}); d={k:payload[k] for k in FEATURES if k in payload}
 try:
  base,e=predict(d)
  if e:return jsonify(error=e),503
  altered=d.copy();altered.update(changes);after,e=predict(altered)
  if e:return jsonify(error=e),503
  return jsonify(original=round(base,2),simulated=round(after,2),difference=round(after-base,2))
 except Exception as ex:return jsonify(error=str(ex)),400
@app.post('/api/explain')
def explain():
 d=request.get_json(force=True)
 try:
  base,e=predict(d)
  if e:return jsonify(error=e),503
  impacts=[]
  # Local perturbation sensitivity: model response when one input changes; not SHAP attribution.
  deltas={'study_hours':1.0,'attendance':5.0,'previous_marks':5.0,'assignment_score':5.0,'internal_marks':5.0,'test_scores':5.0,'sleep_hours':1.0,'internet_access':1,'class_participation':1.0,'historical_performance':5.0}
  limits={'study_hours':(0,24),'attendance':(0,100),'previous_marks':(0,100),'assignment_score':(0,100),'internal_marks':(0,100),'test_scores':(0,100),'sleep_hours':(0,24),'internet_access':(0,1),'class_participation':(0,10),'historical_performance':(0,100)}
  for k in FEATURES:
   alt=d.copy();lo,hi=limits[k];alt[k]=min(hi,max(lo,float(alt[k])+deltas[k]))
   val,e=predict(alt)
   impacts.append({'feature':k,'baseline':round(base,2),'perturbed':round(val,2),'change':round(val-base,2),'step':deltas[k]})
  impacts.sort(key=lambda x:abs(x['change']),reverse=True)
  return jsonify(base_prediction=round(base,2),method='Local one-feature perturbation sensitivity (not SHAP)',impacts=impacts)
 except Exception as ex:return jsonify(error=str(ex)),400
@app.post('/api/subjects')
def subjects():
 marks=request.get_json(force=True).get('subject_marks',{})
 if not marks:return jsonify(error='Provide subject_marks object'),400
 vals={k:float(v) for k,v in marks.items()}
 if any(not 0<=v<=100 for v in vals.values()):return jsonify(error='Marks must be between 0 and 100'),400
 weak=[k for k,v in vals.items() if v<60];strong=[k for k,v in vals.items() if v>=80]
 return jsonify(subjects=vals,average=round(sum(vals.values())/len(vals),2),weak=weak,strong=strong,suggestions=[f'Practice and seek faculty guidance in {s}' for s in weak])
@app.get('/api/history')
def history():
 with db() as c:rows=c.execute('SELECT id,data,predicted,category,risk,created_at FROM records ORDER BY id DESC LIMIT 100').fetchall()
 return jsonify([dict(r) for r in rows])
@app.post('/api/upload')
def upload():
 f=request.files.get('file')
 if not f:return jsonify(error='CSV file required'),400
 try:
  df=pd.read_csv(f);return jsonify(rows=len(df),columns=list(df.columns),message='CSV parsed; batch mapping is a future step.')
 except Exception as ex:return jsonify(error=str(ex)),400
if __name__=='__main__':app.run(host='0.0.0.0',port=5000,debug=True)

# Role-based authentication. Student registration is public; faculty requires an invite code; admin is provisioned from environment variables.
import secrets
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
app.config['SECRET_KEY']=os.environ.get('SECRET_KEY') or secrets.token_hex(32)
TOKEN_SERIALIZER=URLSafeTimedSerializer(app.config['SECRET_KEY'],salt='edupredict-auth')
def auth_token(user_id): return TOKEN_SERIALIZER.dumps({'user_id':user_id})
def current_user():
 h=request.headers.get('Authorization','')
 if not h.startswith('Bearer '): return None
 try:
  data=TOKEN_SERIALIZER.loads(h[7:],max_age=60*60*24*7)
  with db() as c: row=c.execute('SELECT id,name,email,role FROM users WHERE id=? AND active=1',(data['user_id'],)).fetchone()
  return dict(row) if row else None
 except (BadSignature,SignatureExpired,KeyError): return None
def require_roles(*roles):
 def deco(fn):
  @wraps(fn)
  def wrapped(*args,**kwargs):
   u=current_user()
   if not u:return jsonify(error='Authentication required'),401
   if roles and u['role'] not in roles:return jsonify(error='You are not authorized for this action'),403
   request.current_user=u
   return fn(*args,**kwargs)
  return wrapped
 return deco
with db() as c:
 c.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN (\'student\',\'faculty\',\'admin\')),active INTEGER NOT NULL DEFAULT 1,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)')
 admin_email=os.environ.get('ADMIN_EMAIL');admin_password=os.environ.get('ADMIN_PASSWORD')
 if admin_email and admin_password:
  c.execute('INSERT OR IGNORE INTO users(name,email,password_hash,role) VALUES(?,?,?,?)',('System Administrator',admin_email.lower(),generate_password_hash(admin_password),'admin'))
@app.post('/api/auth/register')
def auth_register():
 d=request.get_json(force=True) or {}; name=str(d.get('name','')).strip();email=str(d.get('email','')).strip().lower();password=str(d.get('password',''));role=d.get('role','student')
 if role not in ('student','faculty'):return jsonify(error='Only student and faculty self-registration is allowed'),403
 if not name or len(name)>100 or '@' not in email or len(email)>254:return jsonify(error='Enter a valid name and email'),400
 if len(password)<8:return jsonify(error='Password must contain at least 8 characters'),400
 if role=='faculty' and (not os.environ.get('FACULTY_INVITE_CODE') or not secrets.compare_digest(str(d.get('invite_code','')),os.environ['FACULTY_INVITE_CODE'])):return jsonify(error='A valid faculty invitation code is required'),403
 try:
  with db() as c:c.execute('INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)',(name,email,generate_password_hash(password),role))
  return jsonify(message='Registration successful'),201
 except sqlite3.IntegrityError:return jsonify(error='An account with this email already exists'),409
@app.post('/api/auth/login')
def auth_login():
 d=request.get_json(force=True) or {};email=str(d.get('email','')).strip().lower();password=str(d.get('password',''));role=d.get('role')
 with db() as c: u=c.execute('SELECT id,name,email,password_hash,role FROM users WHERE email=? AND active=1',(email,)).fetchone()
 if not u or not check_password_hash(u['password_hash'],password):return jsonify(error='Invalid email or password'),401
 if role!=u['role']:return jsonify(error='This account does not have the selected role'),403
 user={'id':u['id'],'name':u['name'],'email':u['email'],'role':u['role']}
 return jsonify(user=user,token=auth_token(u['id']))
@app.get('/api/auth/me')
@require_roles('student','faculty','admin')
def auth_me():return jsonify(user=request.current_user)
@app.post('/api/auth/users')
@require_roles('admin')
def admin_create_user():
 d=request.get_json(force=True) or {};role=d.get('role');name=str(d.get('name','')).strip();email=str(d.get('email','')).strip().lower();password=str(d.get('password',''))
 if role not in ('faculty','admin') or not name or '@' not in email or len(password)<8:return jsonify(error='Provide valid name, email, password and faculty/admin role'),400
 try:
  with db() as c:c.execute('INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)',(name,email,generate_password_hash(password),role))
  return jsonify(message=f'{role.title()} account created'),201
 except sqlite3.IntegrityError:return jsonify(error='Email already exists'),409
