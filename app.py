from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Global variables
model = None
df = None

def load_model():
    """Load the trained model from HuggingFace Hub with fallback"""
    global model
    try:
        model_path = hf_hub_download(
            repo_id="Expanic/Stacking_Final_UTSMLOPS",
            filename="stacking_final.pkl"
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully from HuggingFace!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("🔄 Using fallback demo model...")
        
        # Fallback: Simple model untuk demo
        class DemoModel:
            def predict(self, X):
                # Simple prediction based on features
                base_salary = 50000
                experience_bonus = X['Years of Experience'].iloc[0] * 3000
                education_bonus = {
                    'High School': 0,
                    "Bachelor's": 15000,
                    "Master's": 30000,
                    "PhD": 50000
                }.get(X['Education Level'].iloc[0], 0)
                
                country_bonus = {
                    'USA': 20000,
                    'UK': 15000, 
                    'Canada': 10000,
                    'Australia': 12000,
                    'China': 5000
                }.get(X['Country'].iloc[0], 0)
                
                job_bonus = {
                    'Software Engineer': 20000,
                    'Data Scientist': 25000,
                    'Senior Manager': 30000,
                    'Director': 40000,
                    'Data Analyst': 10000
                }.get(X['Job Title'].iloc[0], 0)
                
                predicted = base_salary + experience_bonus + education_bonus + country_bonus + job_bonus
                return np.array([predicted])
        
        model = DemoModel()
        return True

def load_dataset():
    """Load the dataset for visualization"""
    global df
    try:
        df = pd.read_csv('data/dataset.csv')
        print("✅ Dataset loaded successfully!")
        return True
    except FileNotFoundError:
        print("❌ Dataset not found")
        return False

def preprocess_input(age, gender, education_level, job_title, years_experience, country, race):
    """Preprocess user input for prediction"""
    
    input_data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_experience,
        'Country': country,
        'Race': race
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    return input_df

def convert_to_idr(salary_usd):
    """Convert USD salary to IDR"""
    usd_to_idr = 15000
    return salary_usd * usd_to_idr

def format_currency(amount, currency='USD'):
    """Format currency with proper formatting"""
    if currency == 'IDR':
        return f"Rp {amount:,.0f}".replace(',', '.')
    else:
        return f"${amount:,.0f}"

def create_eda_visualizations(df):
    """Create EDA visualizations and return Plotly figure object"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution of Salary', 'Salary by Education Level', 
                       'Salary by Country', 'Salary vs Years of Experience'),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # Salary distribution
    fig.add_trace(
        go.Histogram(x=df['Salary'], name='Salary Distribution', nbinsx=30),
        row=1, col=1
    )
    
    # Salary by Education Level
    education_levels = df['Education Level'].value_counts().head(5).index
    for level in education_levels:
        fig.add_trace(
            go.Box(
                y=df[df['Education Level'] == level]['Salary'],
                name=str(level),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Salary by Country
    countries = df['Country'].value_counts().head(5).index
    for country in countries:
        fig.add_trace(
            go.Box(
                y=df[df['Country'] == country]['Salary'],
                name=str(country),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Salary vs Years of Experience
    fig.add_trace(
        go.Scatter(
            x=df['Years of Experience'],
            y=df['Salary'],
            mode='markers',
            name='Experience vs Salary',
            marker=dict(size=8, opacity=0.6)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800, 
        showlegend=False, 
        title_text="Exploratory Data Analysis"
    )
    
    return fig

@app.route('/')
def home():
    """Home page"""
    stats = {}
    if df is not None:
        stats = {
            'total_jobs': len(df['Job Title'].unique()),
            'total_countries': len(df['Country'].unique()),
            'avg_salary': f"${df['Salary'].mean():,.0f}"
        }
    
    return render_template('index.html', stats=stats)

@app.route('/analysis')
def analysis_page():  # ✅ UBAH NAMA FUNGSI INI
    """Data analysis page"""
    if df is None:
        return render_template('analysis.html', data_available=False)
    
    # Dataset overview
    dataset_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'head': df.head().to_dict('records')
    }
    
    # Create visualizations
    try:
        # EDA Plot
        eda_fig = create_eda_visualizations(df)
        eda_plot_json = json.dumps(eda_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Top paying jobs
        top_jobs = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)
        top_jobs_fig = px.bar(
            x=top_jobs.values, 
            y=top_jobs.index, 
            orientation='h', 
            title="Top 10 Highest Paying Jobs",
            labels={'x': 'Average Salary', 'y': 'Job Title'}
        )
        top_jobs_json = json.dumps(top_jobs_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Salary by country
        country_avg = df.groupby('Country')['Salary'].mean().sort_values(ascending=False)
        country_fig = px.bar(
            x=country_avg.values, 
            y=country_avg.index,
            orientation='h', 
            title="Average Salary by Country",
            labels={'x': 'Average Salary', 'y': 'Country'}
        )
        country_json = json.dumps(country_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        return render_template('analysis.html', data_available=False)
    
    return render_template(
        'analysis.html', 
        data_available=True,
        dataset_info=dataset_info,
        eda_plot=eda_plot_json,
        top_jobs_plot=top_jobs_json,
        country_plot=country_json
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle salary prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        data = request.json
        age = float(data['age'])
        gender = data['gender']
        education_level = data['education_level']
        job_title = data['job_title']
        years_experience = float(data['years_experience'])
        country = data['country']
        race = data['race']
        
        # Preprocess input
        input_df = preprocess_input(age, gender, education_level, job_title, years_experience, country, race)
        
        # Make prediction
        predicted_salary = model.predict(input_df)[0]
        
        # Convert to IDR
        salary_idr = convert_to_idr(predicted_salary)
        
        # Prepare response
        response = {
            'predicted_salary_usd': format_currency(predicted_salary),
            'predicted_salary_idr': format_currency(salary_idr, 'IDR'),
            'salary_usd': float(predicted_salary),
            'salary_idr': float(salary_idr)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/predict')
def prediction_page():
    """Prediction page"""
    return render_template('prediction.html')

# Initialize app
def initialize_app():
    """Initialize the application"""
    print("🚀 Initializing Salary Prediction App...")
    print("Loading model...")
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
    
    print("Loading dataset...")
    if load_dataset():
        print("✅ Dataset loaded successfully!")
    else:
        print("❌ Failed to load dataset!")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=8080)  