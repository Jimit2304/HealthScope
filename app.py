# ==============================================================================
# HealthScope Backend (Full) – Auth + History + Metrics + Pages (MySQL Version)
# ==============================================================================
import os, json, datetime, csv, io
from typing import Dict, Any

from flask import Flask, request, jsonify, g, session, render_template, redirect, url_for, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

import mysql.connector
from mysql.connector import Error

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# ------------------ Config ------------------
APP_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(APP_DIR, 'diabetes[1].csv')
STATIC_DIR = os.path.join(APP_DIR, 'static')

# MySQL connection settings
DB_CONFIG = {
    "host": "localhost",
    "user": "root",         # change if different
    "password": "root",     # change if different
    "database": "diapredict",
    "port": 3306,
    "charset": "utf8mb4"
}

FEATURES_ALL = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
SERVING_FEATURES = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']
TARGET_COL = 'Outcome'

# Health ranges for validation and feedback
HEALTH_RANGES = {
    'glucose': {'normal': (70, 99), 'prediabetic': (100, 125), 'diabetic': (126, 300)},
    'bmi': {'underweight': (0, 18.5), 'normal': (18.5, 24.9), 'overweight': (25, 29.9), 'obese': (30, 50)},
    'blood_pressure': {'normal': (60, 120), 'elevated': (120, 129), 'high': (130, 200)}
}

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY','dev-secret-key')
CORS(app, supports_credentials=True)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

os.makedirs(STATIC_DIR, exist_ok=True)

model: Pipeline = None
train_summary: Dict[str, Any] = {}
feature_stats: Dict[str, Any] = {}
diagnostics: Dict[str, Any] = {}

# ------------------ DB ------------------
def get_db():
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(**DB_CONFIG)
        except Error as e:
            print(f"Database connection failed: {e}")
            raise RuntimeError(f"Cannot connect to database: {e}")
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        role ENUM('user','admin') NOT NULL DEFAULT 'user'
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        input_data JSON NOT NULL,
        prediction TINYINT NOT NULL,
        probability FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    # Ensure role column exists for older databases
    try:
        cursor.execute("SHOW COLUMNS FROM users LIKE 'role'")
        has_role = cursor.fetchone() is not None
        if not has_role:
            cursor.execute("ALTER TABLE users ADD COLUMN role ENUM('user','admin') NOT NULL DEFAULT 'user'")
            db.commit()
    except Exception:
        # Ignore errors here; login logic computes role from username as fallback
        db.rollback()
    
    # Allow NULL user_id in predictions table for deleted users
    try:
        cursor.execute("DESCRIBE predictions")
        columns = cursor.fetchall()
        user_id_nullable = False
        for col in columns:
            if col[0] == 'user_id' and col[2] == 'YES':
                user_id_nullable = True
                break
        
        if not user_id_nullable:
            print("Making user_id column nullable...")
            cursor.execute("ALTER TABLE predictions MODIFY COLUMN user_id INT NULL")
            db.commit()
            print("user_id column is now nullable")
    except Exception as e:
        print(f"Error modifying predictions table: {e}")
        db.rollback()
    db.commit()
    cursor.close()

# ------------------ Auth Helpers ------------------
def current_user_id():
    return session.get('user_id')

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user_id():
            return redirect(url_for('login_page'))
        return fn(*args, **kwargs)
    return wrapper

# ------------------ ML Training ------------------
def train_and_evaluate():
    global model, train_summary, feature_stats, diagnostics
    if not os.path.exists(DATA_FILE):
        raise RuntimeError('Dataset not found')
    df = pd.read_csv(DATA_FILE)
    df6 = df.drop(['SkinThickness','DiabetesPedigreeFunction'], axis=1)

    for col in ['Glucose','BloodPressure','Insulin','BMI']:
        df6.loc[df6[col]==0, col] = df6[col].mean()

    X = df6.drop(TARGET_COL, axis=1)
    y = df6[TARGET_COL]

    feature_stats = {'mean': X.mean().to_dict(), 'std': X.std(ddof=0).replace(0,1.0).to_dict()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob>=0.5).astype(int)

    train_summary = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }

    # Confusion Matrix data (no plotting)
    cm = confusion_matrix(y_test, y_pred)
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    
    # cache diagnostics for dynamic charts
    diagnostics = {
        'confusion_matrix': cm.astype(int).tolist(),
        'roc': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }

# ------------------ Pages ------------------
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

# Convenience routes to avoid 404s when users visit common variants
@app.route('/index.html')
@app.route('/home')
def home_alias():
    return redirect(url_for('home_page'))

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/history')
@login_required
def history_page():
    return render_template('history.html')

@app.route('/stats')
@login_required
def stats_page():
    if session.get('role') != 'admin':
        return redirect(url_for('home_page'))
    return render_template('stats.html')

@app.route('/dashboard')
@login_required
def dashboard_page():
    return render_template('dashboard.html')



# ------------------ API ------------------
@app.get('/api/health')
def health():
    return jsonify({'status':'ok'})

@app.get('/api/me')
def me():
    uid = current_user_id()
    if not uid:
        return jsonify({'authenticated': False})
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, username, role FROM users WHERE id=%s", (uid,))
    row = cursor.fetchone()
    cursor.close()
    if not row:
        session.clear()
        return jsonify({'authenticated': False})
    return jsonify({'authenticated': True, 'user': row})

@app.post('/api/register')
def api_register():
    data = request.get_json(force=True)
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    # Role rule: username 'admin12' is admin, all others are user
    role = 'admin' if username.lower() == 'admin12' else 'user'
    if not username or not password:
        return jsonify({'error':'username and password required'}), 400
    db = get_db()
    cursor = db.cursor()
    try:
        try:
            cursor.execute("INSERT INTO users(username, password_hash, role) VALUES (%s,%s,%s)",
                           (username, generate_password_hash(password), role))
        except Error:
            # Fallback for older schema without role column
            cursor.execute("INSERT INTO users(username, password_hash) VALUES (%s,%s)",
                           (username, generate_password_hash(password)))
        db.commit()
        return jsonify({'status':'ok'})
    except Error:
        return jsonify({'error': 'username already exists'}), 400
    finally:
        cursor.close()

@app.post('/api/login')
def api_login():
    data = request.get_json(force=True)
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, password_hash, role FROM users WHERE username=%s", (username,))
    row = cursor.fetchone()
    cursor.close()
    if not row or not check_password_hash(row['password_hash'], password):
        return jsonify({'error':'invalid credentials'}), 401
    session['user_id'] = row['id']
    # Enforce role rule regardless of DB value
    session['role'] = 'admin' if username.lower() == 'admin12' else 'user'
    return jsonify({'status':'ok'})

@app.post('/api/logout')
def api_logout():
    session.clear()
    return jsonify({'status':'ok'})

@app.get('/api/metrics')
def api_metrics():
    if not train_summary:
        return jsonify({'error':'metrics unavailable'}), 500
    return jsonify(train_summary)

@app.get('/api/model-info')
def api_model_info():
    if model is None:
        return jsonify({'error':'model not initialized'}), 500
    clf = model.named_steps['model']
    return jsonify({
        'algorithm':'LogisticRegression',
        'features': SERVING_FEATURES,
        'coef': clf.coef_.ravel().tolist(),
        'intercept': float(clf.intercept_[0]),
        'feature_means': feature_stats.get('mean', {}),
        'feature_stds': feature_stats.get('std', {})
    })

@app.get('/api/diagnostics')
def api_diagnostics():
    if not diagnostics:
        return jsonify({'error':'diagnostics unavailable'}), 500
    return jsonify(diagnostics)

@app.get('/api/health-ranges')
def api_health_ranges():
    return jsonify(HEALTH_RANGES)

def analyze_risk_factors(inputs):
    risks = []
    glucose = inputs.get('Glucose', 0)
    bmi = inputs.get('BMI', 0)
    bp = inputs.get('BloodPressure', 0)
    age = inputs.get('Age', 0)
    family_diabetes = inputs.get('FamilyDiabetes', 0)
    
    if glucose >= 126:
        risks.append({'factor': 'Glucose', 'level': 'High', 'message': 'Glucose levels indicate diabetes risk'})
    elif glucose >= 100:
        risks.append({'factor': 'Glucose', 'level': 'Moderate', 'message': 'Glucose in prediabetic range'})
    
    if bmi >= 30:
        risks.append({'factor': 'BMI', 'level': 'High', 'message': 'BMI indicates obesity'})
    elif bmi >= 25:
        risks.append({'factor': 'BMI', 'level': 'Moderate', 'message': 'BMI indicates overweight'})
    
    if bp >= 130:
        risks.append({'factor': 'Blood Pressure', 'level': 'High', 'message': 'High blood pressure detected'})
    
    if age >= 45:
        risks.append({'factor': 'Age', 'level': 'Moderate', 'message': 'Age increases diabetes risk'})
    
    if family_diabetes >= 2:
        risks.append({'factor': 'Family History', 'level': 'High', 'message': 'Strong family history of diabetes'})
    elif family_diabetes >= 1:
        risks.append({'factor': 'Family History', 'level': 'Moderate', 'message': 'Family history of diabetes present'})
    
    return risks

@app.post('/api/predict')
@login_required
def api_predict():
    if model is None:
        return jsonify({'error':'model not initialized'}), 500
    try:
        d = request.get_json(force=True)
        req = ['pregnancies','glucose','blood_pressure','insulin','bmi','age','family_diabetes']
        if any(k not in d for k in req):
            return jsonify({'error':'missing fields'}), 400
        x = {
            'Pregnancies': int(d['pregnancies']),
            'Glucose': float(d['glucose']),
            'BloodPressure': float(d['blood_pressure']),
            'Insulin': float(d['insulin']),
            'BMI': float(d['bmi']),
            'Age': float(d['age']),
            'FamilyDiabetes': int(d['family_diabetes'])
        }

        # Match training-time preprocessing: replace zeros for select features
        for k in ['Glucose','BloodPressure','Insulin','BMI']:
            if x[k] == 0 and 'mean' in feature_stats and k in feature_stats['mean']:
                x[k] = float(feature_stats['mean'][k])

        # Use DataFrame with explicit columns to avoid any ordering mismatch
        row_df = pd.DataFrame([x], columns=SERVING_FEATURES)
        proba = float(model.predict_proba(row_df)[0,1])
        pred = int(proba >= 0.5)

        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO predictions(user_id, input_data, prediction, probability) VALUES (%s,%s,%s,%s)",
                       (current_user_id(), json.dumps(x), pred, proba))
        db.commit()
        cursor.close()

        risk_factors = analyze_risk_factors(x)
        return jsonify({
            'prediction': pred, 
            'probability': proba,
            'risk_factors': risk_factors,
            'confidence': 'High' if abs(proba - 0.5) > 0.3 else 'Medium' if abs(proba - 0.5) > 0.15 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': f'bad request: {e}'}), 400

@app.get('/api/history')
@login_required
def api_history():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    is_admin = (session.get('role') == 'admin')
    if is_admin:
        cursor.execute("""
            SELECT p.id, p.input_data, p.prediction, p.probability, p.created_at, u.username
            FROM predictions p
            LEFT JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
        """)
    else:
        cursor.execute("""
            SELECT p.id, p.input_data, p.prediction, p.probability, p.created_at, u.username
            FROM predictions p
            LEFT JOIN users u ON u.id = p.user_id
            WHERE p.user_id=%s
            ORDER BY p.created_at DESC
        """, (current_user_id(),))
    rows = cursor.fetchall()
    cursor.close()
    items = []
    for r in rows:
        items.append({
            'id': r['id'],
            'username': r.get('username') or 'Deleted User',
            'inputs': json.loads(r['input_data']),
            'prediction': r['prediction'],
            'probability': r['probability'],
            'created_at': r['created_at'].strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({'items': items, 'is_admin': is_admin})

@app.get('/api/export/history')
@login_required
def export_history():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    is_admin = (session.get('role') == 'admin')
    if is_admin:
        cursor.execute("""
            SELECT p.input_data, p.prediction, p.probability, p.created_at, u.username
            FROM predictions p LEFT JOIN users u ON u.id = p.user_id ORDER BY p.created_at DESC
        """)
    else:
        cursor.execute("""
            SELECT p.input_data, p.prediction, p.probability, p.created_at, u.username
            FROM predictions p LEFT JOIN users u ON u.id = p.user_id WHERE p.user_id=%s ORDER BY p.created_at DESC
        """, (current_user_id(),))
    rows = cursor.fetchall()
    cursor.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Username', 'Pregnancies', 'Glucose', 'Blood Pressure', 'Insulin', 'BMI', 'Age', 'Family Diabetes', 'Prediction', 'Probability'])
    
    for r in rows:
        inputs = json.loads(r['input_data'])
        writer.writerow([
            r['created_at'].strftime('%Y-%m-%d %H:%M:%S'), r['username'] or 'Deleted User',
            inputs.get('Pregnancies', ''), inputs.get('Glucose', ''), inputs.get('BloodPressure', ''),
            inputs.get('Insulin', ''), inputs.get('BMI', ''), inputs.get('Age', ''),
            inputs.get('FamilyDiabetes', ''), 'Diabetic' if r['prediction'] else 'Non-Diabetic', f"{r['probability']:.3f}"
        ])
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=diabetes_predictions.csv'
    return response

@app.get('/api/dashboard/stats')
@login_required
def dashboard_stats():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    uid = current_user_id()
    is_admin = (session.get('role') == 'admin')
    
    if is_admin:
        # Admin sees all users' data
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total_predictions = cursor.fetchone()['total']
        
        cursor.execute("SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction")
        prediction_breakdown = {str(r['prediction']): r['count'] for r in cursor.fetchall()}
        
        cursor.execute("""
            SELECT DATE(created_at) as date, AVG(probability) as avg_risk
            FROM predictions WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at) ORDER BY date
        """)
        trend_data = [{'date': r['date'].strftime('%Y-%m-%d'), 'risk': float(r['avg_risk'])} for r in cursor.fetchall()]
        
        # Additional admin stats
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute("""
            SELECT u.id, u.username, COUNT(p.id) as prediction_count
            FROM users u LEFT JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id, u.username ORDER BY prediction_count DESC LIMIT 10
        """)
        top_users = [{'id': r['id'], 'username': r['username'], 'count': r['prediction_count']} for r in cursor.fetchall()]
        
    else:
        # Regular user sees only their data
        cursor.execute("SELECT COUNT(*) as total FROM predictions WHERE user_id=%s", (uid,))
        total_predictions = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT prediction, COUNT(*) as count FROM predictions WHERE user_id=%s GROUP BY prediction
        """, (uid,))
        prediction_breakdown = {str(r['prediction']): r['count'] for r in cursor.fetchall()}
        
        cursor.execute("""
            SELECT DATE(created_at) as date, AVG(probability) as avg_risk
            FROM predictions WHERE user_id=%s AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at) ORDER BY date
        """, (uid,))
        trend_data = [{'date': r['date'].strftime('%Y-%m-%d'), 'risk': float(r['avg_risk'])} for r in cursor.fetchall()]
        
        total_users = None
        top_users = []
    
    cursor.close()
    return jsonify({
        'total_predictions': total_predictions, 
        'prediction_breakdown': prediction_breakdown, 
        'trend_data': trend_data,
        'is_admin': is_admin,
        'total_users': total_users,
        'top_users': top_users
    })

@app.post('/api/batch-predict')
@login_required
def batch_predict():
    if model is None:
        return jsonify({'error':'model not initialized'}), 500
    
    try:
        data = request.get_json(force=True)
        predictions_data = data.get('predictions', [])
        
        if not predictions_data:
            return jsonify({'error': 'No prediction data provided'}), 400
        
        results = []
        db = get_db()
        cursor = db.cursor()
        
        for item in predictions_data:
            req = ['pregnancies','glucose','blood_pressure','insulin','bmi','age','family_diabetes']
            if any(k not in item for k in req):
                continue
                
            x = {
                'Pregnancies': int(item['pregnancies']),
                'Glucose': float(item['glucose']),
                'BloodPressure': float(item['blood_pressure']),
                'Insulin': float(item['insulin']),
                'BMI': float(item['bmi']),
                'Age': float(item['age']),
                'FamilyDiabetes': int(item['family_diabetes'])
            }
            
            for k in ['Glucose','BloodPressure','Insulin','BMI']:
                if x[k] == 0 and 'mean' in feature_stats and k in feature_stats['mean']:
                    x[k] = float(feature_stats['mean'][k])
            
            row_df = pd.DataFrame([x], columns=SERVING_FEATURES)
            proba = float(model.predict_proba(row_df)[0,1])
            pred = int(proba >= 0.5)
            risk_factors = analyze_risk_factors(x)
            
            cursor.execute("INSERT INTO predictions(user_id, input_data, prediction, probability) VALUES (%s,%s,%s,%s)",
                          (current_user_id(), json.dumps(x), pred, proba))
            
            results.append({'inputs': x, 'prediction': pred, 'probability': proba, 'risk_factors': risk_factors})
        
        db.commit()
        cursor.close()
        return jsonify({'results': results, 'count': len(results)})
        
    except Exception as e:
        return jsonify({'error': f'batch prediction failed: {e}'}), 400

@app.delete('/api/prediction/<int:prediction_id>')
@login_required
def delete_prediction(prediction_id):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    is_admin = (session.get('role') == 'admin')
    
    if is_admin:
        cursor.execute("SELECT user_id FROM predictions WHERE id=%s", (prediction_id,))
    else:
        cursor.execute("SELECT user_id FROM predictions WHERE id=%s AND user_id=%s", (prediction_id, current_user_id()))
    
    row = cursor.fetchone()
    if not row:
        cursor.close()
        return jsonify({'error': 'Prediction not found or access denied'}), 404
    
    cursor.execute("DELETE FROM predictions WHERE id=%s", (prediction_id,))
    db.commit()
    cursor.close()
    return jsonify({'status': 'ok'})

@app.get('/api/users')
@login_required
def get_users():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, username, role FROM users ORDER BY username")
    users = cursor.fetchall()
    cursor.close()
    return jsonify({'users': users})

@app.delete('/api/user/<int:user_id>')
@login_required
def delete_user(user_id):
    if session.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    db = get_db()
    cursor = db.cursor(dictionary=True)
    
    try:
        # Check if user exists
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            cursor.close()
            return jsonify({'error': 'User not found'}), 404
        
        cursor.execute("UPDATE predictions SET user_id = NULL WHERE user_id = %s", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        db.commit()
        cursor.close()
        return jsonify({'status': 'ok'})
        
    except Exception as e:
        db.rollback()
        cursor.close()
        return jsonify({'error': f'Failed to delete user: {str(e)}'}), 500

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# --------------- Main ---------------
if __name__ == '__main__':
    try:
        with app.app_context():
            init_db()
            train_and_evaluate()
        
        local_ip = get_local_ip()
        port = 5000
        
        print("\n" + "="*50)
        print("🚀 DiaPredict Server Started!")
        print("="*50)
        print(f"🔗 Universal Link: http://{local_ip}:{port}")
        print("\n📱 Use this link on any device:")
        print("   • Mobile phones")
        print("   • Laptops")
        print("   • Tablets")
        print("   • Desktop computers")
        print("\n⚠️  All devices must be on same WiFi")
        print("="*50)
        
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("\n🔧 Check: MySQL server, database, credentials")