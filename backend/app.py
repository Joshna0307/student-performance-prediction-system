import os, json
import psycopg
from psycopg.rows import dict_row
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, pandas as pd
app=Flask(__name__); CORS(app)
MODEL_PATH='model/ann.keras'; SCALER_PATH='model/scaler.joblib'
FEATURES=['study_hours','attendance','previous_marks','assignment_score','internal_marks','test_scores','sleep_hours','internet_access','class_participation','historical_performance']
def db():
  url=os.environ.get('DATABASE_URL')
  if not url: raise RuntimeError('DATABASE_URL is not configured')
  return psycopg.connect(url, row_factory=dict_row)
def init_db():
  with db() as c:
   c.execute('CREATE TABLE IF NOT EXISTS students(id BIGSERIAL PRIMARY KEY, name TEXT, email TEXT UNIQUE)')
   c.execute('CREATE TABLE IF NOT EXISTS records(id BIGSERIAL PRIMARY KEY, student_id BIGINT, data TEXT, predicted REAL, category TEXT, risk TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
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
   u=current_user()
   if not u:return jsonify(error='Please sign in again to save your assessment'),401
   with db() as c:c.execute('INSERT INTO records(user_id,data,predicted,category,risk) VALUES(%s,%s,%s,%s,%s)',(u['id'],json.dumps(d),p,cat,r))
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
  u=current_user()
  if not u:return jsonify(error='Authentication required'),401
  with db() as c:
   if u['role']=='student':
    rows=c.execute("SELECT r.id,u.name AS name,'' AS roll,r.predicted AS predicted_marks,r.category,r.risk,r.created_at FROM records r JOIN users u ON u.id=r.user_id WHERE r.user_id=%s ORDER BY r.id DESC LIMIT 100",(u['id'],)).fetchall()
   elif u['role']=='faculty':
    rows=c.execute("SELECT r.id,u.name AS name,'' AS roll,r.predicted AS predicted_marks,r.category,r.risk,r.created_at FROM records r JOIN users u ON u.id=r.user_id WHERE u.role='student' ORDER BY r.id DESC LIMIT 200").fetchall()
   else:
    rows=c.execute("SELECT r.id,u.name AS name,'' AS roll,r.predicted AS predicted_marks,r.category,r.risk,r.created_at FROM records r LEFT JOIN users u ON u.id=r.user_id ORDER BY r.id DESC LIMIT 200").fetchall()
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
   with db() as c: row=c.execute('SELECT id,name,email,role FROM users WHERE id=%s AND active=TRUE',(data['user_id'],)).fetchone()
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

@app.post('/api/attendance/bulk')
@require_roles('faculty','admin')
def save_attendance_bulk():
  d=request.get_json(force=True) or {}
  course=str(d.get('course','')).strip();day=str(d.get('attendance_date','')).strip();items=d.get('attendance',[])
  if not course or len(course)>120 or not day or not isinstance(items,list) or not items:
   return jsonify(error='Provide course, date and the full class attendance list'),400
  try:
   clean=[]
   for item in items:
    sid=int(item.get('student_id'));status=str(item.get('status','')).title()
    if status not in ('Present','Absent'):return jsonify(error='Each student must be marked Present or Absent'),400
    clean.append((sid,course,day,status,request.current_user['id']))
   with db() as c:
    ids=[x[0] for x in clean]
    valid=c.execute("SELECT id FROM users WHERE role='student' AND active=TRUE AND id=ANY(%s)",(ids,)).fetchall()
    if len(valid)!=len(set(ids)):return jsonify(error='One or more student accounts are invalid'),400
    for row in clean:
     c.execute("INSERT INTO attendance(student_id,course,attendance_date,status,marked_by) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(student_id,course,attendance_date) DO UPDATE SET status=EXCLUDED.status,marked_by=EXCLUDED.marked_by,updated_at=CURRENT_TIMESTAMP",row)
   return jsonify(message=f"Attendance saved for {len(clean)} students",saved_count=len(clean)),200
  except (TypeError,ValueError):return jsonify(error='Invalid student ID or date'),400

with db() as c:
 c.execute('CREATE TABLE IF NOT EXISTS users(id BIGSERIAL PRIMARY KEY,name TEXT NOT NULL,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN (\'student\',\'faculty\',\'admin\')),active BOOLEAN NOT NULL DEFAULT TRUE,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
 c.execute('ALTER TABLE records ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE SET NULL')
 admin_email=os.environ.get('ADMIN_EMAIL');admin_password=os.environ.get('ADMIN_PASSWORD')
 if admin_email and admin_password:
  c.execute('INSERT INTO users(name,email,password_hash,role) VALUES(%s,%s,%s,%s) ON CONFLICT(email) DO UPDATE SET name=EXCLUDED.name,password_hash=EXCLUDED.password_hash,role=EXCLUDED.role,active=TRUE',('System Administrator',admin_email.lower(),generate_password_hash(admin_password),'admin'))
@app.post('/api/auth/register')
def auth_register():
 d=request.get_json(force=True) or {}; name=str(d.get('name','')).strip();email=str(d.get('email','')).strip().lower();password=str(d.get('password',''));role=d.get('role','student')
 if role not in ('student','faculty'):return jsonify(error='Only student and faculty self-registration is allowed'),403
 if not name or len(name)>100 or '@' not in email or len(email)>254:return jsonify(error='Enter a valid name and email'),400
 if len(password)<8:return jsonify(error='Password must contain at least 8 characters'),400
 if role=='faculty' and (not os.environ.get('FACULTY_INVITE_CODE') or not secrets.compare_digest(str(d.get('invite_code','')),os.environ['FACULTY_INVITE_CODE'])):return jsonify(error='A valid faculty invitation code is required'),403
 try:
  with db() as c:c.execute('INSERT INTO users(name,email,password_hash,role) VALUES(%s,%s,%s,%s)',(name,email,generate_password_hash(password),role))
  return jsonify(message='Registration successful'),201
 except psycopg.errors.UniqueViolation:return jsonify(error='An account with this email already exists'),409
@app.post('/api/auth/login')
def auth_login():
 d=request.get_json(force=True) or {};email=str(d.get('email','')).strip().lower();password=str(d.get('password',''));role=d.get('role')
 with db() as c: u=c.execute('SELECT id,name,email,password_hash,role FROM users WHERE email=%s AND active=TRUE',(email,)).fetchone()
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
  with db() as c:c.execute('INSERT INTO users(name,email,password_hash,role) VALUES(%s,%s,%s,%s)',(name,email,generate_password_hash(password),role))
  return jsonify(message=f'{role.title()} account created'),201
 except psycopg.errors.UniqueViolation:return jsonify(error='Email already exists'),409


@app.get('/api/admin/overview')
@require_roles('admin')
def admin_overview():
 with db() as c:
  counts=c.execute("SELECT role,COUNT(*) AS total FROM users WHERE active=TRUE GROUP BY role").fetchall()
  recent=c.execute("SELECT id,name,email,role,created_at FROM users WHERE active=TRUE ORDER BY id DESC").fetchall()
 totals={'student':0,'faculty':0,'admin':0}
 for row in counts: totals[row['role']]=row['total']
 return jsonify(counts=totals,total=sum(totals.values()),users=[dict(x) for x in recent])

@app.get('/api/faculty/students')
@require_roles('faculty','admin')
def faculty_students():
 with db() as c:
  rows=c.execute("SELECT id,name,email,created_at FROM users WHERE role='student' AND active=TRUE ORDER BY lower(name)").fetchall()
 return jsonify(students=[dict(x) for x in rows],total=len(rows))


# Faculty-controlled attendance; students have read-only access.
with db() as c:
 c.execute("CREATE TABLE IF NOT EXISTS attendance(id BIGSERIAL PRIMARY KEY, student_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, course TEXT NOT NULL, attendance_date DATE NOT NULL, status TEXT NOT NULL CHECK(status IN ('Present','Absent','Late')), marked_by BIGINT NOT NULL REFERENCES users(id), updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(student_id,course,attendance_date))")

@app.get('/api/attendance')
@require_roles('student','faculty','admin')
def get_attendance():
 u=request.current_user
 with db() as c:
  if u['role']=='student':
   rows=c.execute("SELECT a.id,a.student_id,u.name AS student_name,a.course,a.attendance_date,a.status,a.updated_at FROM attendance a JOIN users u ON u.id=a.student_id WHERE a.student_id=%s ORDER BY a.attendance_date DESC",(u['id'],)).fetchall()
  else:
   rows=c.execute("SELECT a.id,a.student_id,u.name AS student_name,a.course,a.attendance_date,a.status,a.updated_at,m.name AS marked_by_name FROM attendance a JOIN users u ON u.id=a.student_id JOIN users m ON m.id=a.marked_by ORDER BY a.attendance_date DESC,u.name",()).fetchall()
 return jsonify(attendance=[dict(x) for x in rows])

@app.post('/api/attendance')
@require_roles('faculty','admin')
def save_attendance():
 d=request.get_json(force=True) or {}
 try:
  sid=int(d.get('student_id'));course=str(d.get('course','')).strip();day=str(d.get('attendance_date','')).strip();status=str(d.get('status','')).title()
  if not course or len(course)>120 or status not in ('Present','Absent','Late'):return jsonify(error='Enter course and valid attendance status'),400
  with db() as c:
   student=c.execute("SELECT id FROM users WHERE id=%s AND role='student' AND active=TRUE",(sid,)).fetchone()
   if not student:return jsonify(error='Student account not found'),404
   row=c.execute("INSERT INTO attendance(student_id,course,attendance_date,status,marked_by) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(student_id,course,attendance_date) DO UPDATE SET status=EXCLUDED.status,marked_by=EXCLUDED.marked_by,updated_at=CURRENT_TIMESTAMP RETURNING id,student_id,course,attendance_date,status",(sid,course,day,status,request.current_user['id'])).fetchone()
  return jsonify(message='Attendance saved successfully',attendance=dict(row)),201
 except (TypeError,ValueError):return jsonify(error='Select a valid student and date'),400


# Faculty-created quizzes and server-graded student attempts.
with db() as c:
 c.execute("""CREATE TABLE IF NOT EXISTS quizzes(
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  subject TEXT NOT NULL,
  instructions TEXT NOT NULL DEFAULT '',
  questions JSONB NOT NULL,
  created_by BIGINT NOT NULL REFERENCES users(id),
  published BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
 )""")
 c.execute("""CREATE TABLE IF NOT EXISTS quiz_attempts(
  id BIGSERIAL PRIMARY KEY,
  quiz_id BIGINT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  student_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  answers JSONB NOT NULL,
  score INTEGER NOT NULL,
  total INTEGER NOT NULL,
  submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
 )""")

@app.get('/api/quizzes')
@require_roles('student','faculty','admin')
def list_quizzes():
 u=request.current_user
 with db() as c:
  if u['role']=='student':
   rows=c.execute("SELECT id,title,subject,instructions,questions,created_at FROM quizzes WHERE published=TRUE ORDER BY created_at DESC").fetchall()
  else:
   rows=c.execute("SELECT q.id,q.title,q.subject,q.instructions,q.questions,q.published,q.created_at,u.name AS faculty_name FROM quizzes q JOIN users u ON u.id=q.created_by ORDER BY q.created_at DESC").fetchall()
 out=[]
 for row in rows:
  item=dict(row); item['questions']=[{k:v for k,v in q.items() if k!='correct_index'} for q in item['questions']]
  out.append(item)
 return jsonify(quizzes=out)

@app.post('/api/quizzes')
@require_roles('faculty','admin')
def create_quiz():
 d=request.get_json(force=True) or {}
 title=str(d.get('title','')).strip();subject=str(d.get('subject','')).strip();instructions=str(d.get('instructions','')).strip();questions=d.get('questions')
 if not title or len(title)>160 or not subject or len(subject)>120 or not isinstance(questions,list) or not 1<=len(questions)<=50:
  return jsonify(error='Enter a title, subject and between 1 and 50 questions'),400
 clean=[]
 for q in questions:
  prompt=str(q.get('prompt','')).strip();options=q.get('options');correct=q.get('correct_index')
  if not prompt or not isinstance(options,list) or len(options)!=4 or any(not str(x).strip() for x in options) or not isinstance(correct,int) or correct not in range(4):
   return jsonify(error='Each question needs text, four options and one correct answer'),400
  clean.append({'prompt':prompt,'options':[str(x).strip() for x in options],'correct_index':correct})
 with db() as c:
  row=c.execute("INSERT INTO quizzes(title,subject,instructions,questions,created_by) VALUES(%s,%s,%s,%s::jsonb,%s) RETURNING id,title,subject,created_at",(title,subject,instructions,json.dumps(clean),request.current_user['id'])).fetchone()
 return jsonify(message='Quiz created and published',quiz=dict(row)),201

@app.post('/api/quizzes/<int:quiz_id>/submit')
@require_roles('student')
def submit_quiz(quiz_id):
 d=request.get_json(force=True) or {};answers=d.get('answers')
 if not isinstance(answers,list):return jsonify(error='Submit an answer for each question'),400
 with db() as c:
  quiz=c.execute("SELECT id,questions FROM quizzes WHERE id=%s AND published=TRUE",(quiz_id,)).fetchone()
  if not quiz:return jsonify(error='Quiz not found'),404
  questions=quiz['questions']
  if len(answers)!=len(questions):return jsonify(error='Please answer every question'),400
  score=sum(1 for i,q in enumerate(questions) if isinstance(answers[i],int) and answers[i]==q['correct_index'])
  row=c.execute("INSERT INTO quiz_attempts(quiz_id,student_id,answers,score,total) VALUES(%s,%s,%s::jsonb,%s,%s) RETURNING id,submitted_at",(quiz_id,request.current_user['id'],json.dumps(answers),score,len(questions))).fetchone()
 return jsonify(message='Quiz submitted successfully',score=score,total=len(questions),percentage=round(score/len(questions)*100),attempt_id=row['id'],submitted_at=row['submitted_at'].isoformat()),201

@app.get('/api/quiz-results')
@require_roles('student','faculty','admin')
def quiz_results():
 u=request.current_user
 with db() as c:
  if u['role']=='student':
   rows=c.execute("SELECT a.id,q.title,q.subject,a.score,a.total,a.submitted_at FROM quiz_attempts a JOIN quizzes q ON q.id=a.quiz_id WHERE a.student_id=%s ORDER BY a.submitted_at DESC",(u['id'],)).fetchall()
  else:
   rows=c.execute("SELECT a.id,q.title,q.subject,u.name AS student_name,a.score,a.total,a.submitted_at FROM quiz_attempts a JOIN quizzes q ON q.id=a.quiz_id JOIN users u ON u.id=a.student_id ORDER BY a.submitted_at DESC LIMIT 500").fetchall()
 return jsonify(results=[dict(x) for x in rows])
