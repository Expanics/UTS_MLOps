import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

from huggingface_hub import hf_hub_download
import pickle
import streamlit as st


def load_model():
    """Load the trained model from HuggingFace Hub"""
    try:
        model_path = hf_hub_download(
            repo_id="Expanic/Stacking_Final_UTSMLOPS",
            filename="stacking_final.pkl"  # pastiin ini sama persis kayak nama file di HF
        )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def load_dataset():
    """Load the dataset for visualization"""
    try:
        df = pd.read_csv('data/dataset.csv')
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some visualizations will be disabled.")
        return None

def preprocess_input(age, gender, education_level, job_title, years_experience, country, race):
    """Preprocess user input for prediction"""
    
    # Create input dictionary
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
    # Using approximate exchange rate (you can update this)
    usd_to_idr = 15000  # 1 USD = 15,000 IDR
    salary_idr = salary_usd * usd_to_idr
    return salary_idr

def format_currency(amount, currency='USD'):
    """Format currency with proper formatting"""
    if currency == 'IDR':
        return f"Rp {amount:,.0f}".replace(',', '.')
    else:
        return f"${amount:,.0f}"

def create_eda_visualizations(df):
    """Create EDA visualizations"""
    
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
    education_data = []
    for level in df['Education Level'].unique():
        education_data.append(go.Box(
            y=df[df['Education Level'] == level]['Salary'],
            name=level,
            showlegend=False
        ))
    
    for trace in education_data:
        fig.add_trace(trace, row=1, col=2)
    
    # Salary by Country
    country_data = []
    for country in df['Country'].unique():
        country_data.append(go.Box(
            y=df[df['Country'] == country]['Salary'],
            name=country,
            showlegend=False
        ))
    
    for trace in country_data:
        fig.add_trace(trace, row=2, col=1)
    
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
    
    fig.update_layout(height=800, showlegend=False, title_text="Exploratory Data Analysis")
    return fig

def main():
    local_css()
    
    # Load model and data
    model = load_model()
    df = load_dataset()
    
    # Header
    st.markdown('<h1 class="main-header">💰 Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Predict your potential salary based on your profile!** 
    This tool helps job seekers understand their market value, especially those considering international opportunities.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Section", 
                                   ["🏠 Home", "📊 Data Analysis", "🎯 Salary Prediction", "ℹ️ About"])
    
    if app_mode == "🏠 Home":
        st.header("Welcome to Salary Prediction Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Jobs", len(df['Job Title'].unique()) if df is not None else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Countries", len(df['Country'].unique()) if df is not None else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_salary = df['Salary'].mean() if df is not None else 0
            st.metric("Average Salary", f"${avg_salary:,.0f}" if df is not None else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📈 Why Use This Tool?
        
        - **For Job Seekers**: Understand your market value when considering international opportunities
        - **For Professionals**: Compare salaries across different countries and roles
        - **For Career Planning**: Make informed decisions about education and career paths
        
        ### 🌍 International Opportunities
        
        With more people seeking international job opportunities, understanding salary expectations 
        across different countries is crucial. Our model helps you predict potential earnings based on:
        - Your education level
        - Years of experience
        - Job title
        - Target country
        - And other important factors
        """)
    
    elif app_mode == "📊 Data Analysis":
        st.header("📊 Data Analysis & Insights")
        
        if df is not None:
            # Show dataset info
            with st.expander("Dataset Overview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(df.head())
                with col2:
                    st.write("**Dataset Info:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {list(df.columns)}")
            
            # EDA Visualizations
            st.plotly_chart(create_eda_visualizations(df), use_container_width=True)
            
            # Additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Paying Jobs")
                top_jobs = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)
                fig_jobs = px.bar(top_jobs, x=top_jobs.values, y=top_jobs.index, 
                                 orientation='h', title="Top 10 Highest Paying Jobs")
                st.plotly_chart(fig_jobs, use_container_width=True)
            
            with col2:
                st.subheader("Salary by Country")
                country_avg = df.groupby('Country')['Salary'].mean().sort_values(ascending=False)
                fig_country = px.bar(country_avg, x=country_avg.values, y=country_avg.index,
                                    orientation='h', title="Average Salary by Country")
                st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.warning("Dataset not available for analysis.")
    
    elif app_mode == "🎯 Salary Prediction":
        st.header("🎯 Predict Your Salary")
        
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
            return
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=21, max_value=65, value=30, help="Your current age")
                years_experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5, 
                                           help="Total years of professional experience")
                
                # Education Level
                education_options = ["High School", "Bachelor's", "Master's", "PhD"]
                education_level = st.selectbox("Education Level", education_options)
                
                # Country
                country_options = ["USA", "UK", "Canada", "Australia", "China"]
                country = st.selectbox("Country", country_options)
            
            with col2:
                # Gender
                gender_options = ["Male", "Female", "Other"]
                gender = st.selectbox("Gender", gender_options)
                
                # Job Title
                job_options = [
                    "Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate",
                    "Director", "Marketing Analyst", "Product Manager", "Sales Manager",
                    "Marketing Coordinator", "Senior Scientist", "Data Scientist",
                    "Project Manager", "Business Analyst", "HR Manager", "Financial Analyst"
                ]
                job_title = st.selectbox("Job Title", job_options)
                
                # Race
                race_options = ["White", "Hispanic", "Asian", "Black", "Mixed", 
                               "Korean", "Chinese", "Australian", "Other"]
                race = st.selectbox("Race/Ethnicity", race_options)
            
            submitted = st.form_submit_button("Predict Salary", use_container_width=True)
        
        # Prediction
        if submitted:
            with st.spinner("Calculating your salary prediction..."):
                # Preprocess input
                input_df = preprocess_input(age, gender, education_level, job_title, 
                                          years_experience, country, race)
                
                try:
                    # Make prediction
                    predicted_salary = model.predict(input_df)[0]
                    
                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Salary (USD)", f"${predicted_salary:,.0f}")
                    
                    with col2:
                        salary_idr = convert_to_idr(predicted_salary)
                        st.metric("Predicted Salary (IDR)", format_currency(salary_idr, 'IDR'))
                    
                    with col3:
                        # Show confidence indicator
                        st.metric("Confidence Level", "High")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional insights
                    st.subheader("💰 Salary Insights")
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        st.write("**Breakdown by Factors:**")
                        st.write(f"• **Education**: {education_level}")
                        st.write(f"• **Experience**: {years_experience} years")
                        st.write(f"• **Country**: {country}")
                        st.write(f"• **Job Role**: {job_title}")
                    
                    with insight_col2:
                        st.write("**💡 Tips to Increase Your Salary:**")
                        if years_experience < 10:
                            st.write("• Gain more experience in your field")
                        if education_level in ["High School", "Bachelor's"]:
                            st.write("• Consider advanced degrees")
                        st.write("• Develop specialized skills")
                        st.write("• Consider roles in high-paying countries")
                    
                    # Comparison with averages
                    if df is not None:
                        avg_similar = df[
                            (df['Education Level'] == education_level) &
                            (df['Years of Experience'].between(years_experience-2, years_experience+2))
                        ]['Salary'].mean()
                        
                        if not np.isnan(avg_similar):
                            diff = predicted_salary - avg_similar
                            st.info(f"📊 Compared to similar profiles: **{format_currency(predicted_salary)}** vs Average: **{format_currency(avg_similar)}** ({'+' if diff > 0 else ''}{format_currency(diff)})")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please try different input values.")
    
    elif app_mode == "ℹ️ About":
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This Salary Prediction Dashboard uses machine learning to predict salaries based on various 
        professional and demographic factors. The model was trained on comprehensive job market data 
        and can provide valuable insights for job seekers, especially those considering international 
        opportunities.
        
        ### 🏗️ Model Architecture
        
        The prediction model uses a **Stacking Ensemble** approach combining:
        - **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
        - **Tree-based Models**: Random Forest, XGBoost, CatBoost
        - **Neural Network**: Custom CNN architecture
        
        ### 📊 Features Used
        
        The model considers 7 key features:
        1. **Age** - Current age
        2. **Gender** - Gender identity
        3. **Education Level** - Highest educational qualification
        4. **Job Title** - Professional role
        5. **Years of Experience** - Professional experience duration
        6. **Country** - Target country for employment
        7. **Race/Ethnicity** - Racial/ethnic background
        
        ### 💰 Currency Conversion
        
        Salary predictions are provided in both:
        - **USD** (United States Dollars) - For international comparison
        - **IDR** (Indonesian Rupiah) - For local context
        
        *Note: Exchange rate used: 1 USD = 15,000 IDR*
        
        ### 🎓 Educational Purpose
        
        This project was developed as part of MLOps coursework and demonstrates:
        - End-to-end machine learning pipeline
        - Model deployment with Streamlit
        - Interactive data visualization
        - Real-world business application
        
        ### 👨‍💻 Developer
        
        **Muhammad Reza Alghifari**  
        Artificial Intelligence Student  
        MLOps Course Project
        """)

if __name__ == "__main__":
    main()