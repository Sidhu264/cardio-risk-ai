from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os
import joblib
import cv2
import numpy as np
import sqlite3
import matplotlib
from flask_bcrypt import Bcrypt  # For password hashing
# Set the backend before importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import base64


from flask import make_response
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image



app = Flask(__name__)
bcrypt = Bcrypt(app)  # Initialize Bcrypt for password hashing
app.secret_key = "your_secret_key_here"  # Required for flash messages

UPLOAD_FOLDER = "uploads"
DATABASE = "heart_risk.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Load model
MODEL_PATH = "model/adaboost_resnet_model.pkl"
model = joblib.load(MODEL_PATH)

# Health Metrics & Risk Labels
risk_labels = {
    0: "Age = 20-25 , SBP = 111-126 mmHg, DBP = 80-85 mmHg, BMI = 18-25, HbA1c = 5.4 -7.0, No Risk - You are Healthy",
    1: "Age = 30-35 , SBP = 140-160 mmHg, DBP = 80-90 mmHg, BMI = 27-29, HbA1c = 4-5.6, Very Low Risk - 20%",
    2: "Age = 35-40 , SBP = 150-166 mmHg, DBP = 85-95 mmHg, BMI = 29-31, HbA1c = 7.5 -10.5, Mild Risk - 40%",
    3: "Age = 40-50 , SBP = 155-170 mmHg, DBP = 90-100 mmHg, BMI = 28-33, HbA1c = 10.0 -12.5, Moderate Risk - 60%",
    4: "Age = 45-60 , SBP = 160-176 mmHg, DBP = 95-100 mmHg, BMI = 30-35, HbA1c = 13.4 -14.9, High Risk"
}

risk_output = {
    0: {"Age": "20-25","SBP": "111-126", "DBP": "80-85", "BMI": "18-25", "HbA1c": "5.4 -7.0", "Risk Score": "No Risk - You are Healthy -0"},
    1: {"Age": "30-35", "SBP": "140-160", "DBP": "80-90", "BMI": "27-29", "HbA1c": "4-5.6", "Risk Score": 20},
    2: {"Age": "35-40", "SBP": "150-166", "DBP": "85-95", "BMI": "29-31", "HbA1c": "7.5 -10.5", "Risk Score": 40},
    3: {"Age": "40-45", "SBP": "155-170", "DBP": "90-100", "BMI": "28-33", "HbA1c": "10.0 -12.5", "Risk Score": 60},
    4: {"Age": "45-60", "SBP": "160-176", "DBP": "95-100", "BMI": "30-35", "HbA1c": "13.4 -14.9", "Risk Score": 80}
}

risk_classes = {
    0: {"SBP": 118, "DBP": 83, "BMI": 22, "HbA1c": 6.2, "Risk Score": 0},
    1: {"SBP": 150, "DBP": 85, "BMI": 28, "HbA1c": 5.2, "Risk Score": 20},
    2: {"SBP": 158, "DBP": 90, "BMI": 30, "HbA1c": 8.5, "Risk Score": 40},
    3: {"SBP": 165, "DBP": 95, "BMI": 32, "HbA1c": 10.2, "Risk Score": 60},
    4: {"SBP": 172, "DBP": 100, "BMI": 34, "HbA1c": 14, "Risk Score": 80}
}  

ideal_values = {"SBP": 120, "DBP": 80, "BMI": 24, "HbA1c": 5.5, "Risk Score": 0}   

precautions = {
    0: """
    <b>No Risk - Healthy</b><br>
    
        <li><b>Dietary Recommendations:</b> Adhere to a nutritionally balanced diet rich in phytonutrients, antioxidants, and essential macronutrients. Prioritize whole foods, including leafy greens, cruciferous vegetables, and lean protein sources such as poultry, fish, and legumes. Minimize intake of processed foods, refined sugars, and trans fats.</li>
        <li><b>Physical Activity:</b> Engage in moderate-intensity aerobic exercise for a minimum of 150 minutes per week to maintain optimal cardiovascular health and body composition. Incorporate resistance training twice weekly to preserve muscle mass and metabolic efficiency.</li>
        <li><b>Hydration:</b> Maintain adequate hydration with a daily water intake of 2-3 liters, adjusted for individual factors such as body weight and activity level.</li>
        <li><b>Preventive Care:</b> Schedule annual comprehensive health evaluations, including lipid profiling, fasting glucose, and blood pressure monitoring, to ensure early detection of potential health deviations.</li>
    
    """,

    1: """
    ⚠️ <b>Very Low Risk - 20%</b><br>
    
        <li><b>Dietary Modifications:</b> Restrict sodium intake to <2,300 mg/day to mitigate hypertension risk. Emphasize consumption of potassium-rich foods (e.g., bananas, spinach) to counterbalance sodium effects. Incorporate soluble fiber (e.g., oats, psyllium) to improve glycemic control and lipid metabolism.</li>
        <li><b>Physical Activity:</b> Perform low-impact aerobic exercises such as brisk walking or stationary cycling for 30 minutes daily, 5 days per week. Monitor heart rate to ensure exercise intensity remains within 50-70% of maximum heart rate.</li>
        <li><b>Glycemic Monitoring:</b> Regularly assess HbA1c levels (target range: 4-5.6%) and fasting glucose to identify early signs of insulin resistance.</li>
        <li><b>Hydration:</b> Ensure consistent hydration with electrolyte-balanced fluids to support renal and cardiovascular function.</li>
    """,

    2: """
    🟡 <b>Mild Risk - 40%</b><br>
        <li><b>Exercise Regimen:</b> Incorporate moderate-intensity physical activities such as jogging, swimming, or yoga for 30-45 minutes, 5 times per week. Focus on improving cardiovascular endurance and reducing visceral adiposity.</li>
        <li><b>Lifestyle Modifications:</b> Abstain from tobacco use and limit alcohol consumption to <1 standard drink per day to reduce cardiovascular and hepatic strain.</li>
        <li><b>Blood Pressure Management:</b> Conduct daily blood pressure monitoring (target: <130/80 mmHg) and maintain a log for clinical review. Schedule cardiovascular risk assessments, including lipid panels and carotid Doppler studies, every 3-6 months.</li>
    """,

    3: """
    🟠 <b>Moderate Risk - 60%</b><br>
        <li><b>Dietary Restrictions:</b> Implement a strict low-sodium diet (<1,500 mg/day) and eliminate processed foods. Increase intake of omega-3 fatty acids (e.g., salmon, walnuts, flaxseeds) to reduce systemic inflammation and improve endothelial function.</li>
        <li><b>Exercise Protocol:</b> Combine strength training (2-3 sessions/week) with moderate-to-high-intensity cardio (e.g., HIIT, running) to optimize BMI (target: 28-33) and enhance insulin sensitivity.</li>
        <li><b>Glycemic Control:</b> Perform frequent blood glucose monitoring (fasting target: 70-130 mg/dL) and adjust dietary intake or pharmacotherapy as needed.</li>
        <li><b>Medical Follow-Up:</b> Maintain regular consultations with a cardiologist or endocrinologist for advanced risk stratification and therapeutic adjustments.</li>
    """,

    4: """
    🚨 <b>High Risk - 80%+</b><br>
        <li><b>Immediate Intervention:</b> Seek urgent medical evaluation for comprehensive risk assessment and initiation of evidence-based pharmacotherapy (e.g., statins, antihypertensives, antidiabetics).</li>
        <li><b>Dietary Compliance:</b> Adopt a therapeutic lifestyle changes (TLC) diet, strictly avoiding high-cholesterol foods, saturated fats, and refined carbohydrates. Prioritize plant-based, low-glycemic-index foods.</li>
        <li><b>Monitoring:</b> Perform daily blood pressure and glucose monitoring (HbA1c target: <7%). Utilize continuous glucose monitoring (CGM) systems if indicated.</li>
        <li><b>Weight Management:</b> Enroll in a medically supervised weight loss program incorporating caloric restriction, behavioral therapy, and pharmacologic agents (e.g., GLP-1 agonists) if necessary.</li>
        <li><b>Medication Adherence:</b> Ensure strict compliance with prescribed medications and attend regular follow-ups to monitor therapeutic efficacy and adjust treatment plans.</li>
    """
}

# Database Initialization
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            sbp TEXT,
            dbp TEXT,
            bmi TEXT,
            hba1c TEXT,
            age TEXT,
            risk_score INTEGER,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()

init_db()

# Image Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Generate Graphs
def generate_graphs(user_metrics, class_label):
    metrics = list(user_metrics.keys())
    user_values = list(user_metrics.values())
    ideal_values_list = [ideal_values[m] for m in metrics]
    
    # Graph 1: User Metrics
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(metrics, user_values, color="blue", alpha=0.7, label="User Metrics")
    ax1.set_ylabel("Value")
    ax1.set_title(f"Predicted Class {class_label}: {risk_labels[class_label]}")
    ax1.legend()
    img1 = BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    plt.close(fig1)  # Explicitly close the figure
    img1.seek(0)
    graph1_url = base64.b64encode(img1.getvalue()).decode()
    
    # Graph 2: Comparison of User Metrics vs. Ideal Metrics (Grouped Bar Chart)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    # Plot User Metrics and Ideal Metrics side by side
    rects1 = ax2.bar(x - width/2, user_values, width, color="blue", label="User Metrics")
    rects2 = ax2.bar(x + width/2, ideal_values_list, width, color="green", label="Ideal Metrics")

    # Add labels, title, and custom x-axis tick labels
    ax2.set_ylabel("Value")
    ax2.set_title("Comparison: User vs. Ideal Health Metrics")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f"{height}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha="center", va="bottom")

    # Add value labels to the bars
    autolabel(rects1)
    autolabel(rects2)

    # Save the plot to a BytesIO object
    img2 = BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    plt.close(fig2)  # Explicitly close the figure
    img2.seek(0)
    graph2_url = base64.b64encode(img2.getvalue()).decode()
    
    return graph1_url, graph2_url

@app.route('/')
def home():
    return render_template('login.html')

from flask import request

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    # Get data from the form
    name = request.form.get('name')
    age = request.form.get('age')
    sbp = request.form.get('sbp')
    dbp = request.form.get('dbp')
    bmi = request.form.get('bmi')
    hba1c = request.form.get('hba1c')
    risk_score = request.form.get('risk_score')
    graph1 = request.form.get('graph1')
    graph2 = request.form.get('graph2')
    precautions = request.form.get('precautions')

    # Create a BytesIO buffer to store the PDF
    buffer = BytesIO()

    # Create the PDF object
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add Patient Name
    story.append(Paragraph(f"<b>Patient Name:</b> {name}", styles["Title"]))
    story.append(Spacer(1, 12))

    # Add Health Metrics
    story.append(Paragraph("<b>Health Metrics:</b>", styles["Heading2"]))
    story.append(Paragraph(f"<b>Age:</b> {age} years", styles["BodyText"]))
    story.append(Paragraph(f"<b>Systolic BP:</b> {sbp} mmHg", styles["BodyText"]))
    story.append(Paragraph(f"<b>Diastolic BP:</b> {dbp} mmHg", styles["BodyText"]))
    story.append(Paragraph(f"<b>BMI:</b> {bmi} kg/m²", styles["BodyText"]))
    story.append(Paragraph(f"<b>HbA1c:</b> {hba1c}%", styles["BodyText"]))
    story.append(Paragraph(f"<b>Risk Score:</b> {risk_score}%", styles["Heading2"]))
    story.append(Spacer(1, 12))

    # Add Graphs Side by Side
    story.append(Paragraph("<b>Graphs:</b>", styles["Heading2"]))
    graph1_img = Image(BytesIO(base64.b64decode(graph1)), width=250, height=200)
    graph2_img = Image(BytesIO(base64.b64decode(graph2)), width=250, height=200)
    story.append(graph1_img)
    story.append(Spacer(1, 12))
    story.append(graph2_img)
    story.append(Spacer(1, 12))

    # Add Precautions
    story.append(Paragraph("<b>Precautions:</b>", styles["Heading2"]))
    story.append(Paragraph(precautions, styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Build the PDF
    pdf.build(story)

    # Move the buffer's cursor to the beginning
    buffer.seek(0)

    # Create a response with the PDF
    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=patient_report_{name}.pdf"

    return response

@app.route('/dashboard')
def dashboard():
    return render_template('main-dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.check_password_hash(user[3], password):
            return redirect(url_for('dashboard'))  # Redirect to dashboard after successful login
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Please use a different email.", "error")
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    file = request.files['image']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        processed_image = preprocess_image(filepath)
        
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        bottleneck_features = base_model.predict(processed_image)
        risk_class = model.predict(bottleneck_features)[0]
        
        user_metrics = risk_classes[risk_class]
        graph1, graph2 = generate_graphs(user_metrics, risk_class)
        output_risk_metrics = risk_output[risk_class]
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO results (name, sbp, dbp, bmi, hba1c, age, risk_score) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, output_risk_metrics['SBP'], output_risk_metrics['DBP'], output_risk_metrics['BMI'], output_risk_metrics['HbA1c'], output_risk_metrics['Age'], user_metrics['Risk Score']))
        conn.commit()
        conn.close()
        
        return render_template('result-dashboard.html', 
                               name=name, 
                               age=output_risk_metrics['Age'], 
                               sbp=output_risk_metrics['SBP'], 
                               dbp=output_risk_metrics['DBP'], 
                               bmi=output_risk_metrics['BMI'], 
                               hba1c=output_risk_metrics['HbA1c'], 
                               risk_score=output_risk_metrics['Risk Score'], 
                               graph1=graph1, 
                               graph2=graph2,
                               precautions=precautions[risk_class])
    return redirect(url_for('home'))

@app.route('/results')
def results():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results ORDER BY date DESC")
    past_results = cursor.fetchall()
    conn.close()
    return render_template('history.html', past_results=past_results)

if __name__ == '__main__':
    app.run(debug=True)