# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Helper functions
def gpa_to_grouped_letter(gpa):
    """Convert GPA to letter grade with error handling"""
    try:
        gpa = float(gpa)
        if gpa >= 3.7:
            return 'A'
        elif gpa >= 3.0:
            return 'B'
        elif gpa >= 2.0:
            return 'C'
        else:
            return 'D/F'
    except (ValueError, TypeError):
        return 'N/A'

def grade_to_numeric(grade):
    """Convert letter grade to numeric value with error handling"""
    grade_map = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D/F': 1.0}
    return grade_map.get(grade, 0.0)

def generate_recommendations(habits, grade_gap, years_to_grad, stress_level):
    """Generate personalized recommendations based on habits and goals"""
    recommendations = {
        "📚 Study Strategy": [],
        "😴 Wellness": [],
        "⚖️ Balance": []
    }
    
    try:
        # Check for extreme cases first
        if habits['study'] == 0:
            recommendations["📚 Study Strategy"].append(
                "**Critical: No study time detected.** You must allocate time for studying to maintain or improve your grades."
            )
        elif habits['study'] < 2:
            recommendations["📚 Study Strategy"].append(
                "**Warning: Very low study time.** Your grades are likely to drop significantly with less than 2 hours of study per day."
            )
        elif habits['study'] < 6:
            recommendations["📚 Study Strategy"].append(
                "**Increase study time** gradually by 1-2 hours per day"
            )
            
        if habits.get('screen_time', 0) > 3:  # Use get() with default
            recommendations["📚 Study Strategy"].append(
                "**Reduce screen time** during study hours - try the Pomodoro Technique"
            )
        
        # Wellness recommendations
        if habits['sleep'] == 0:
            recommendations["😴 Wellness"].append(
                "**Critical: No sleep detected.** Sleep is essential for learning and memory. Aim for at least 6 hours."
            )
        elif habits['sleep'] < 5:
            recommendations["😴 Wellness"].append(
                "**Warning: Very low sleep.** Your cognitive function is severely impaired with less than 5 hours of sleep."
            )
        elif habits['sleep'] < 7:
            recommendations["😴 Wellness"].append(
                "**Improve sleep habits** - aim for 7-9 hours per night"
            )
            
        if habits['physical'] == 0:
            recommendations["😴 Wellness"].append(
                "**Add some physical activity** - even 15 minutes daily improves focus and reduces stress"
            )
        elif habits['physical'] < 1:
            recommendations["😴 Wellness"].append(
                "**Add 30 minutes of exercise** daily for better focus"
            )
        
        # Balance recommendations
        total_activity = sum(habits.values())
        if total_activity == 0:
            recommendations["⚖️ Balance"].append(
                "**Critical: No activities detected.** You need to engage in some activities to maintain academic performance."
            )
        elif total_activity < 4:
            recommendations["⚖️ Balance"].append(
                "**Warning: Very low activity level.** Your academic performance will likely suffer with minimal daily activities."
            )
        elif total_activity > 16:
            recommendations["⚖️ Balance"].append(
                "**Consider reducing some activities** to prevent burnout"
            )
            
        if habits['social'] == 0:
            recommendations["⚖️ Balance"].append(
                "**Include some social time** - social connections are important for mental health and academic success"
            )
        elif habits['social'] < 1:
            recommendations["⚖️ Balance"].append(
                "**Include some social time** for better stress management"
            )
    except Exception as e:
        recommendations["📚 Study Strategy"].append(
            "**Error generating detailed recommendations.** Please ensure all habits are properly tracked."
        )
    
    return recommendations

def generate_maintenance_tips(habits, stress_level):
    """Generate tips for maintaining current success"""
    tips = [
        "🎯 **Set specific goals** for each study session to maintain focus",
        "📊 **Track your progress** weekly to stay motivated",
        "🧘‍♂️ **Practice stress management** techniques regularly",
        "👥 **Share your strategies** with study groups or classmates",
        "🌱 **Gradually increase challenges** to keep growing"
    ]
    return tips

def generate_critical_warning(habits):
    """Generate critical warnings for extreme cases"""
    warnings = []
    
    # Check for critical issues
    if habits['study'] == 0:
        warnings.append("**CRITICAL: No study time detected.** Your academic performance will likely drop significantly.")
    
    if habits['sleep'] == 0:
        warnings.append("**CRITICAL: No sleep detected.** This will severely impact your cognitive function and learning ability.")
    
    # Check if all activities are zero
    if all(value == 0 for value in habits.values()):
        warnings.append("**CRITICAL: No activities detected.** This pattern will lead to academic failure if not addressed immediately.")
    
    return warnings

# Define the StudentPredictionModel class needed for loading our model
class StudentPredictionModel:
    def __init__(self, model, scaler, grade_encoder, stress_encoder, feature_names):
        self.model = model
        self.scaler = scaler
        self.grade_encoder = grade_encoder
        self.stress_encoder = stress_encoder
        self.grade_mapping = {i: label for i, label in enumerate(grade_encoder.classes_)}
        self.stress_mapping = {i: label for i, label in enumerate(stress_encoder.classes_)}
        self.feature_names = feature_names
    
    def predict(self, X_input):
        """Makes predictions from raw input features and returns human-readable labels"""
        # Engineer features if they're not already present
        X_processed = X_input.copy()
        
        # Create missing features
        if 'Study_Sleep_Interaction' not in X_processed.columns:
            # Study-Sleep interaction
            X_processed['Study_Sleep_Interaction'] = X_processed['Study_Hours_Per_Day'] * X_processed['Sleep_Hours_Per_Day']
            
            # Social-to-Study ratio - handle division by zero
            study_hours = X_processed['Study_Hours_Per_Day'].values[0]
            if study_hours == 0:
                X_processed['Social_Study_Ratio'] = 0  # Default value when study hours is 0
            else:
                X_processed['Social_Study_Ratio'] = X_processed['Social_Hours_Per_Day'] / study_hours
            
            # Total active time
            X_processed['Total_Activity_Hours'] = (
                X_processed['Study_Hours_Per_Day'] + 
                X_processed['Extracurricular_Hours_Per_Day'] + 
                X_processed['Physical_Activity_Hours_Per_Day']
            )
            
            # Study efficiency - handle division by zero
            sleep_hours = X_processed['Sleep_Hours_Per_Day'].values[0]
            if sleep_hours == 0:
                X_processed['Study_Efficiency'] = 0  # Default value when sleep hours is 0
            else:
                X_processed['Study_Efficiency'] = X_processed['Study_Hours_Per_Day'] * (sleep_hours / 8)
            
            # Balance metric - handle division by zero
            if (X_processed['Sleep_Hours_Per_Day'].values[0] == 0 or 
                X_processed['Study_Hours_Per_Day'].values[0] == 0 or 
                X_processed['Physical_Activity_Hours_Per_Day'].values[0] == 0):
                X_processed['Life_Balance'] = 0  # Default value when any component is 0
            else:
                X_processed['Life_Balance'] = (
                    (X_processed['Sleep_Hours_Per_Day'] / 24) * 
                    (X_processed['Study_Hours_Per_Day'] / 24) * 
                    (X_processed['Physical_Activity_Hours_Per_Day'] / 24)
                ) * 100
        
        # Ensure columns are in the right order
        X_processed = X_processed[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make prediction
        y_encoded = self.model.predict(X_scaled)
        
        # Decode predictions
        grade_idx = int(y_encoded[0][0])
        stress_idx = int(y_encoded[0][1])
        
        letter_grade = self.grade_mapping[grade_idx]
        stress_level = self.stress_mapping[stress_idx]
        
        return [letter_grade, stress_level]

# Initialize session state for storing user data
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'current_year': 'Freshman',  # Set default values
        'current_gpa': 3.0,
        'target_graduation_year': datetime.now().year + 4,
        'target_gpa': 3.5,
        'habits_history': [],
        'recommendations_history': []
    }

# Safe model loading with error handling
try:
    model = joblib.load('stacked_multioutput_predictor.pkl')
except Exception as e:
    st.error("Error loading the model. Please ensure the model file exists and is accessible.")
    st.stop()

# App title and description
st.title('🎓 Brain Boost: Academic Success Coach')
st.markdown("""
Welcome to your personalized academic success coach! This AI-powered tool helps you:
- 📈 Track your progress toward your GPA goals
- 💡 Get data-driven recommendations
- 🎯 Simulate the impact of habit changes
- 📊 Monitor your academic journey
""")

# Sidebar for user profile
with st.sidebar:
    st.header("📋 Your Academic Profile")
    
    # Academic Profile Section
    current_year = st.selectbox(
        "Current Year",
        ['Freshman', 'Sophomore', 'Junior', 'Senior'],
        key='year_select',
        help="Your current academic year"
    )
    
    current_gpa = st.number_input(
        "Current GPA",
        min_value=0.0,
        max_value=4.0,
        value=float(st.session_state.user_profile['current_gpa']),
        step=0.01,
        help="Your current cumulative GPA"
    )
    
    target_graduation_year = st.selectbox(
        "Target Graduation Year",
        range(datetime.now().year, datetime.now().year + 6),
        index=4,  # Default to 4 years from now
        help="When do you plan to graduate?"
    )
    
    target_gpa = st.number_input(
        "Target GPA",
        min_value=0.0,
        max_value=4.0,
        value=float(st.session_state.user_profile['target_gpa']),
        step=0.01,
        help="What GPA do you aim to achieve by graduation?"
    )
    
    # Update profile button
    if st.button("Update Profile"):
        st.session_state.user_profile.update({
            'current_year': current_year,
            'current_gpa': current_gpa,
            'target_graduation_year': target_graduation_year,
            'target_gpa': target_gpa
        })
        st.success("Profile updated successfully!")

# Main content area
st.header("📊 Daily Habits Tracker")

# Create tabs for different sections
tabs = st.tabs(["📝 Input Habits", "📈 Progress", "🎯 Recommendations"])

# Input Habits Tab
with tabs[0]:
    st.subheader("Track Your Daily Habits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            help="Average hours spent studying per day"
        )
        
        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.5,
            help="Average hours of sleep per night"
        )
        
        social_hours = st.number_input(
            "Social Activities",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            help="Hours spent on social activities"
        )
    
    with col2:
        physical_activity_hours = st.number_input(
            "Physical Activity",
            min_value=0.0,
            max_value=24.0,
            value=1.5,
            help="Hours spent on exercise and physical activities"
        )
        
        extracurricular_hours = st.number_input(
            "Extracurricular Activities",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            help="Hours spent on clubs, organizations, etc."
        )
        
        screen_time = st.number_input(
            "Screen Time (Entertainment)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            help="Hours spent on entertainment (social media, gaming, etc.)"
        )

    # Calculate remaining hours
    total_tracked_hours = (study_hours + sleep_hours + social_hours + 
                         physical_activity_hours + extracurricular_hours + screen_time)
    remaining_hours = max(0, 24 - total_tracked_hours)  # Ensure non-negative
    
    # Show time allocation
    st.markdown("### ⏰ Time Allocation")
    
    # Fix progress bar to ensure value is between 0 and 1
    progress_value = min(1.0, max(0.0, total_tracked_hours/24))
    st.progress(progress_value)
    
    if total_tracked_hours > 24:
        st.warning("⚠️ Total hours exceed 24 hours. Please adjust your time allocation.")
    else:
        st.caption(f"You have {remaining_hours:.1f} hours unaccounted for in your day")

    # Check for critical cases
    current_habits = {
        'study': float(study_hours),
        'sleep': float(sleep_hours),
        'social': float(social_hours),
        'physical': float(physical_activity_hours),
        'extracurricular': float(extracurricular_hours),
        'screen_time': float(screen_time)
    }
    
    critical_warnings = generate_critical_warning(current_habits)
    if critical_warnings:
        st.error("### ⚠️ Critical Warnings")
        for warning in critical_warnings:
            st.markdown(warning)

    # Analyze and predict button
    if st.button("Analyze My Habits"):
        try:
            # Prepare input features
            input_features = pd.DataFrame({
                'Study_Hours_Per_Day': [study_hours],
                'Extracurricular_Hours_Per_Day': [extracurricular_hours],
                'Sleep_Hours_Per_Day': [sleep_hours],
                'Social_Hours_Per_Day': [social_hours],
                'Physical_Activity_Hours_Per_Day': [physical_activity_hours]
            })
            
            # Get predictions
            prediction = model.predict(input_features)
            predicted_grade = prediction[0]
            predicted_stress = prediction[1]
            
            # Store habits and predictions in history
            st.session_state.user_profile['habits_history'].append({
                'date': datetime.now(),
                'habits': current_habits,
                'predictions': {
                    'grade': predicted_grade,
                    'stress': predicted_stress
                }
            })
            
            # Show success message
            st.success("Analysis complete! Check the Progress and Recommendations tabs for insights.")
            
        except Exception as e:
            st.error("Error analyzing habits. Please check your inputs and try again.")
            st.exception(e)

# Progress Tab
with tabs[1]:
    if st.session_state.user_profile['habits_history']:
        try:
            latest = st.session_state.user_profile['habits_history'][-1]
            predicted_grade = latest['predictions']['grade']
            predicted_stress = latest['predictions']['stress']
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Grade Level",
                    predicted_grade,
                    delta="Target: " + gpa_to_grouped_letter(st.session_state.user_profile['target_gpa'])
                )
            
            with col2:
                st.metric(
                    "Stress Level",
                    predicted_stress
                )
            
            with col3:
                # Calculate time to graduation
                years_to_grad = max(0, st.session_state.user_profile['target_graduation_year'] - 
                                  datetime.now().year)
                st.metric(
                    "Time to Graduation",
                    f"{years_to_grad} years"
                )
            
            # Progress visualization
            if len(st.session_state.user_profile['habits_history']) > 1:
                st.markdown("### 📈 Your Progress")
                try:
                    # Create a DataFrame for visualization
                    history_data = []
                    for entry in st.session_state.user_profile['habits_history']:
                        history_data.append(entry['habits'])
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Show the line chart
                    st.line_chart(history_df)
                    
                    # Add a table view of recent history
                    st.markdown("### 📋 Recent History")
                    st.dataframe(
                        history_df.tail(5).round(2),
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning("Could not display progress chart. Please ensure you have multiple data points.")
        except Exception as e:
            st.error("Error displaying progress. Please try analyzing your habits again.")
    else:
        st.info("Click 'Analyze My Habits' to see your progress!")

# Recommendations Tab
with tabs[2]:
    if st.session_state.user_profile['habits_history']:
        latest = st.session_state.user_profile['habits_history'][-1]
        predicted_grade = latest['predictions']['grade']
        current_habits = latest['habits']
        
        st.markdown("### 🎯 Personalized Action Plan")
        
        # Calculate the gap between current and target
        try:
            current_grade_val = grade_to_numeric(predicted_grade)
            target_grade_val = st.session_state.user_profile['target_gpa']
            grade_gap = target_grade_val - current_grade_val
            
            # Ensure years_to_grad is defined
            years_to_grad = (
                st.session_state.user_profile['target_graduation_year'] - 
                datetime.now().year
            )
            
            # Check for critical cases first
            critical_warnings = generate_critical_warning(current_habits)
            if critical_warnings:
                st.error("### ⚠️ Critical Warnings")
                for warning in critical_warnings:
                    st.markdown(warning)
                
                st.markdown("""
                ### 🚨 Immediate Action Required
                Your current habits indicate a high risk of academic failure. Please make the following changes immediately:
                """)
                
                # Generate specific recommendations
                recommendations = generate_recommendations(
                    current_habits,
                    grade_gap,
                    years_to_grad,
                    latest['predictions']['stress']
                )
                
                # Display recommendations by category
                for category, recs in recommendations.items():
                    if recs:  # Only show categories with recommendations
                        st.markdown(f"#### {category}")
                        for rec in recs:
                            st.markdown(f"- {rec}")
                
                # Add emergency tips
                st.markdown("""
                ### 🆘 Emergency Tips
                - **Start with just 1 hour of study** per day and gradually increase
                - **Prioritize sleep** - aim for at least 6 hours immediately
                - **Seek academic support** from your professors or tutoring center
                - **Consider speaking with an academic advisor** about your situation
                """)
            
            elif grade_gap > 0:
                st.markdown(f"""
                Based on your current habits and goals:
                - You're currently on track for a **{predicted_grade}** grade level
                - Your target GPA of **{st.session_state.user_profile['target_gpa']}** requires a **{gpa_to_grouped_letter(st.session_state.user_profile['target_gpa'])}** grade level
                - To bridge this gap over the next {years_to_grad} years, you'll need to make some adjustments
                """)
                
                # Generate specific recommendations
                recommendations = generate_recommendations(
                    current_habits,
                    grade_gap,
                    years_to_grad,
                    latest['predictions']['stress']
                )
                
                # Display recommendations by category
                for category, recs in recommendations.items():
                    if recs:  # Only show categories with recommendations
                        st.markdown(f"#### {category}")
                        for rec in recs:
                            st.markdown(f"- {rec}")
                
                # Habit change simulation
                st.markdown("### 🔄 Simulate Habit Changes")
                st.markdown("""
                Adjust your habits below to see how they might affect your outcomes.
                Move the sliders to simulate different scenarios.
                """)
                
                # Create simulation sliders with current habits as defaults
                sim_study = st.slider("Study Hours", 0.0, 12.0, float(current_habits['study']))
                sim_sleep = st.slider("Sleep Hours", 4.0, 10.0, float(current_habits['sleep']))
                sim_physical = st.slider("Physical Activity", 0.0, 4.0, float(current_habits['physical']))
                
                if st.button("Simulate Changes"):
                    try:
                        # Prepare simulated input
                        sim_features = pd.DataFrame({
                            'Study_Hours_Per_Day': [sim_study],
                            'Extracurricular_Hours_Per_Day': [current_habits['extracurricular']],
                            'Sleep_Hours_Per_Day': [sim_sleep],
                            'Social_Hours_Per_Day': [current_habits['social']],
                            'Physical_Activity_Hours_Per_Day': [sim_physical]
                        })
                        
                        # Get predictions for simulation
                        sim_prediction = model.predict(sim_features)
                        
                        # Show simulation results
                        st.markdown("#### Simulation Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Current Predicted Grade",
                                predicted_grade
                            )
                        
                        with col2:
                            st.metric(
                                "Simulated Grade",
                                sim_prediction[0],
                                delta=("↑" if sim_prediction[0] > predicted_grade else "↓")
                            )
                        
                        # Show explicit message for grade change
                        if sim_prediction[0] > predicted_grade:
                            st.success("🟢 Your predicted grade will improve!")
                        elif sim_prediction[0] < predicted_grade:
                            st.warning("🔴 Warning: Your predicted grade will drop.")
                        else:
                            st.info("No change in predicted grade.")
                        # Show warning if simulation has critical issues
                        sim_habits = {
                            'study': float(sim_study),
                            'sleep': float(sim_sleep),
                            'social': float(current_habits['social']),
                            'physical': float(sim_physical),
                            'extracurricular': float(current_habits['extracurricular']),
                            'screen_time': float(current_habits['screen_time'])
                        }
                        
                        sim_warnings = generate_critical_warning(sim_habits)
                        if sim_warnings:
                            st.error("### ⚠️ Critical Warnings for Simulation")
                            for warning in sim_warnings:
                                st.markdown(warning)
                    except Exception as e:
                        st.error("Error running simulation. Please try different values.")
            else:
                st.success("""
                🌟 Congratulations! Your current habits are aligned with your academic goals.
                Keep up the great work and consider setting even higher targets!
                """)
                
                st.markdown("### 💪 Maintenance Recommendations")
                maintenance_tips = generate_maintenance_tips(current_habits, latest['predictions']['stress'])
                for tip in maintenance_tips:
                    st.markdown(f"- {tip}")
        except Exception as e:
            st.error("Error calculating recommendations. Please ensure all your profile information is complete.")
    else:
        st.info("Analyze your habits first to get personalized recommendations!")

# Footer
st.markdown("---")
st.caption("✨ Your AI Academic Success Coach - Powered by Machine Learning")


