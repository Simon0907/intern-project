# Student Performance Prediction System

An AI-powered academic analytics platform with user authentication and prediction history tracking.

## Features

- **User Authentication**: Secure login and signup using Supabase
- **Performance Prediction**: Predict student performance based on math, reading, and writing scores
- **Dashboard**: View prediction history and personal statistics
- **Beautiful UI**: Premium glassmorphism design with smooth animations

## Prerequisites

Before running this application, ensure you have:

- Python 3.8 or higher
- Model files:
  - `student_model.joblib`
  - `scaler.joblib`
  - `model_features.joblib`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file contains the Supabase credentials:
```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### First Time Users

1. Click on the "Sign Up" tab
2. Enter your full name, email, and password
3. Click "Sign Up" to create your account
4. Switch to "Login" tab and login with your credentials

### Making Predictions

1. Navigate to the "New Prediction" tab
2. Enter the student's scores:
   - Math Score (0-100)
   - Reading Score (0-100)
   - Writing Score (0-100)
   - Student Gender
3. Click "Predict Performance"
4. View the detailed prediction results with confidence breakdown

### Viewing History

1. Navigate to the "Prediction History" tab
2. View all your past predictions
3. Expand each prediction to see detailed information

### Viewing Statistics

1. Navigate to the "Statistics" tab
2. See your overall statistics including:
   - Total predictions made
   - Average scores across all predictions
   - Most common performance category
   - Performance distribution charts

## Database Schema

The application uses two main tables:

### user_profiles
- Stores user information
- Linked to Supabase auth.users

### predictions
- Stores all prediction history
- Includes scores, predictions, and confidence levels
- Protected by Row Level Security (RLS)

## Security

- All data is protected with Row Level Security
- Users can only access their own data
- Passwords are securely hashed by Supabase
- Authentication tokens are managed automatically

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **ML**: scikit-learn (via joblib)
- **Visualization**: Plotly
