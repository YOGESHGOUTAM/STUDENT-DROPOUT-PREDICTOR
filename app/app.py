import os
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Define the model class (must be identical to the one used in train.py and model.py)
class DropoutPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Define paths relative to the app.py file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# --- Load Model and Preprocessing Components (Cached for Performance) ---
@st.cache_resource # Cache the loading process to run only once
def load_resources():
    try:
        saved_metadata = joblib.load(PREPROCESSOR_PATH)
        pipeline = saved_metadata['pipeline']
        label_encoder = saved_metadata['label_encoder']
        feature_order = saved_metadata['feature_order']
        numerical_cols_for_standardscaler = saved_metadata['numerical_cols']
        categorical_cols_for_onehot = saved_metadata['categorical_cols']
        input_dim = saved_metadata['input_dim']
        best_params = saved_metadata['best_params']
        original_X_dtypes_map = saved_metadata['original_X_dtypes_map']

        # Extract best hyperparameters for model initialization
        hidden_dim = best_params['hidden_dim']
        dropout_rate = best_params['dropout_rate']
        num_classes = len(label_encoder.classes_)

        # Initialize and load the model
        model = DropoutPredictorNet(input_dim, hidden_dim, dropout_rate, num_classes)
        # Ensure mapping to CPU for broader compatibility
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode

        return (pipeline, label_encoder, feature_order,
                numerical_cols_for_standardscaler, categorical_cols_for_onehot,
                model, original_X_dtypes_map)

    except FileNotFoundError:
        st.error(f"Error: Model or preprocessor files not found. "
                 f"Please ensure 'train.py' has been run successfully to generate '{BEST_MODEL_PATH}' and '{PREPROCESSOR_PATH}'.")
        st.stop()
    except KeyError as e:
        st.error(f"Error: Missing key in preprocessor.pkl: {e}. "
                 f"Ensure 'train.py' saves all necessary metadata (pipeline, label_encoder, feature_order, numerical_cols, categorical_cols, input_dim, best_params, original_X_dtypes_map).")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

(pipeline, label_encoder, feature_order,
 numerical_cols_for_standardscaler, categorical_cols_for_onehot,
 model, original_X_dtypes_map) = load_resources()

# --- Prediction Function ---
def predict_dropout(user_data_dict):
    """
    Makes a dropout prediction for a given user's data.

    Args:
        user_data_dict (dict): A dictionary where keys are original feature names (numerical indices 0-35)
                          and values are the user's input for those features.
    Returns:
        tuple: (predicted_label, probabilities_array)
    """
    # 1. Create a DataFrame from user_data_dict. Keys are now numerical indices.
    # Create an empty DataFrame with columns in `feature_order` and then append
    # This approach ensures column order and handles potential missing user inputs for some features
    user_df_raw = pd.DataFrame([user_data_dict])
    user_df_raw = user_df_raw.reindex(columns=feature_order, fill_value=np.nan)


    # 2. Ensure correct data types before transformation based on original_X_dtypes_map
    for col_idx in feature_order: # Iterate through numerical column indices
        original_dtype_str = original_X_dtypes_map.get(col_idx)
        if original_dtype_str == 'object': # This path is unlikely given train.py processes
            user_df_raw[col_idx] = user_df_raw[col_idx].astype(str)
        elif 'int' in original_dtype_str or 'float' in original_dtype_str:
            user_df_raw[col_idx] = pd.to_numeric(user_df_raw[col_idx], errors='coerce')
        # Fill NaNs for numerical columns before preprocessing if any were coerced or missing
        if col_idx in numerical_cols_for_standardscaler and user_df_raw[col_idx].isnull().any():
            # Use median from training data if available, or a sensible default like 0
            # This requires saving the median from the training data during preprocessing
            # For now, using a simple median of current input (or 0 if input is all NaN)
            user_df_raw[col_idx] = user_df_raw[col_idx].fillna(user_df_raw[col_idx].median() if not user_df_raw[col_idx].empty else 0)


    # 3. Transform the user input data using the loaded pipeline
    # The pipeline expects a DataFrame with columns matching training, so X.columns are 0,1,2...35
    transformed = pipeline.transform(user_df_raw)

    # 4. Convert to PyTorch tensor
    input_tensor = torch.tensor(transformed.toarray() if hasattr(transformed, 'toarray') else transformed, dtype=torch.float32)

    # 5. Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # 6. Decode prediction back to original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probabilities.numpy()[0]

# --- Helper Function to Create Input Widgets Dynamically ---
def create_input_widget(feature_idx, original_X_dtypes_map):
    """
    Creates an appropriate Streamlit input widget based on feature type and known mappings.
    Args:
        feature_idx (int): The numerical index of the feature (0 to 35).
        original_X_dtypes_map (dict): A dictionary mapping feature names (numerical indices) to their original dtypes.
    Returns:
        Streamlit widget value.
    """
    # --- CRITICAL: Complete this list with actual feature names from UCI dataset documentation ---
    # The order here must match the column order in your data.csv (0 to 35).
    # THIS IS AN EXAMPLE. YOU NEED TO VERIFY AND COMPLETE ALL 36 FEATURES ACCURATELY.
    feature_names_mapping = {
        0: "Marital status",
        1: "Application mode",
        2: "Course",
        3: "Previous qualification",
        4: "Nationality",
        5: "Mother's qualification",
        6: "Father's qualification",
        7: "Mother's occupation",
        8: "Father's occupation",
        9: "Admission grade",
        10: "Displaced",
        11: "Educational special needs",
        12: "Debtor",
        13: "Tuition fees up to date",
        14: "Scholarship holder",
        15: "Gender", # VERIFY THIS INDEX from docs!
        16: "Age at enrollment",
        17: "International", # VERIFY THIS INDEX from docs!
        18: "Curricular units 1st sem (credited)",
        19: "Curricular units 1st sem (enrolled)",
        20: "Curricular units 1st sem (evaluations)",
        21: "Curricular units 1st sem (approved)",
        22: "Curricular units 1st sem (grade)",
        23: "Curricular units 1st sem (without evaluations)",
        24: "Curricular units 2nd sem (credited)",
        25: "Curricular units 2nd sem (enrolled)",
        26: "Curricular units 2nd sem (evaluations)",
        27: "Curricular units 2nd sem (approved)",
        28: "Curricular units 2nd sem (grade)",
        29: "Curricular units 2nd sem (without evaluations)",
        30: "Unemployment rate",
        31: "Inflation rate",
        32: "GDP",
        33: "Daytime/evening attendance", # Example, verify from docs
        34: "Previous Academic Quota", # Example, verify from docs
        35: "Curricular units (total approved)", # Example, verify from docs
        # Add ALL remaining 36 features (indices 0-35) with their correct, descriptive names.
        # REFER TO UCI DATASET DOCUMENTATION (ID 697) FOR ACCURATE NAMES AND ORDER.
    }
    feature_display_name = feature_names_mapping.get(feature_idx, f"Feature {feature_idx} (Unknown Name)")


    # --- CRITICAL: Complete this with all numerical-to-category mappings from documentation ---
    # The keys are the numerical column INDICES (0, 1, 2, ...), not string names.
    # You MUST verify these indices match the actual columns in your data.csv
    # based on the dataset documentation.
    # THIS IS AN EXAMPLE. YOU NEED TO VERIFY AND COMPLETE ALL MAPPINGS ACCURATELY.
    mappings = {
        # Feature 0: Marital status
        0: {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto Union', 6: 'Legally Separated'},
        # Feature 1: Application mode
        1: {1: '1st phase - general contingent', 2: 'Order-reapplication', 5: '1st phase - special contingent (Azores Island)',
            7: 'Holders of other higher courses', 10: 'Transfer', 15: 'Change of course', 16: 'Transfer from other institution',
            17: 'Change of course (other institution)', 18: 'International student (bachelor)',
            39: '1st phase - special contingent (Madeira Island)', 42: '2nd phase - general contingent',
            43: '2nd phase - special contingent (Azores Island)', 44: '2nd phase - special contingent (Madeira Island)'},
        # Feature 2: Course - This list is extensive, complete it accurately from UCI documentation
        2: {33: 'Agronomy', 171: 'Veterinary Nursing', 9085: 'Communication Design', 9254: 'Social Service (Evening)',
            9119: 'Journalism and Communication', 9147: 'Management', 9500: 'Nursing', 9003: 'Journalism and Communication',
            9670: 'Veterinary Nursing', 9130: 'Basic Education', 9140: 'Biofuel Production Technologies', 9254: 'Social Service',
            9773: 'Tourism', 9123: 'Social Service (Evening)', 9853: 'Management (Evening)', 9801: 'Nursing (Evening)',
            9100: 'Public Administration', 9147: 'Management (Evening)', 9130: 'Basic Education (Evening)',
            # ... ADD ALL OTHER COURSE CODES AND NAMES
        },
        # Feature 3: Previous qualification
        3: {1: 'Secondary education', 2: 'Higher education - bachelors degree', 3: 'Higher education - degree',
            4: 'Higher education - masters', 5: 'Higher education - doctorate', 6: 'Primary education (4th year)',
            9: 'Primary education (1st Cycle - 1st to 4th year)', 10: 'Primary education (2nd Cycle - 5th to 6th year)',
            11: 'Primary education (3rd Cycle - 7th to 9th year)', 12: 'Basic education 3rd Cycle (9th year) equivalent to Portuguese',
            14: 'Specialized higher education', 18: 'Frequency of higher education', 19: '12th year of schooling not completed'
        },
        # Feature 4: Nationality
        4: {1: 'Portuguese', 6: 'German', 11: 'Brazilian', 13: 'Angolan', 14: 'Cape Verdean', 17: 'Spanish',
            21: 'Guinean', 22: 'Mozambican', 25: 'Santomean', 26: 'Turkish', 27: 'Venezuelan', 28: 'Indian',
            32: 'Italian', 41: 'French', 62: 'Lithuanian', 100: 'Moldovan', 101: 'Mexican', 103: 'Nigerian',
            105: 'Dutch', 108: 'Cuban', 109: 'Colombian', 110: 'Belgian'
        },
        # Feature 5: Mother's qualification
        5: {1: 'Secondary education', 2: 'Higher education - bachelors degree', 3: 'Higher education - degree',
            4: 'Higher education - masters', 5: 'Higher education - doctorate', 6: 'Primary education (4th year)',
            9: 'Primary education (1st Cycle - 1st to 4th year)', 10: 'Primary education (2nd Cycle - 5th to 6th year)',
            11: 'Primary education (3rd Cycle - 7th to 9th year)', 12: 'Basic education 3rd Cycle (9th year) equivalent to Portuguese',
            14: 'Specialized higher education', 18: 'Frequency of higher education', 19: '12th year of schooling not completed'
        },
        # Feature 6: Father's qualification
        6: {1: 'Secondary education', 2: 'Higher education - bachelors degree', 3: 'Higher education - degree',
            4: 'Higher education - masters', 5: 'Higher education - doctorate', 6: 'Primary education (4th year)',
            9: 'Primary education (1st Cycle - 1st to 4th year)', 10: 'Primary education (2nd Cycle - 5th to 6th year)',
            11: 'Primary education (3rd Cycle - 7th to 9th year)', 12: 'Basic education 3rd Cycle (9th year) equivalent to Portuguese',
            14: 'Specialized higher education', 18: 'Frequency of higher education', 19: '12th year of schooling not completed'
        },
        # Feature 7: Mother's occupation
        7: {0: 'Student', 1: 'Rural Producer', 2: 'Specialist in Data Processing', 3: 'Specialist in Information Technologies',
            4: 'Management and Administration Technician', 5: 'Hotel, Catering, Trade and Other Services Manager',
            6: 'Accountant', 7: 'Consultant', 8: 'Construction Manager', 9: 'Commercial Manager',
            10: 'Real Estate Agent', 11: 'Lawyer', 12: 'Doctor', 13: 'Nurse', 14: 'Pharmacist',
            15: 'Teacher', 16: 'Bank Employee', 17: 'Public Administration Employee', 18: 'Other Administrative Staff',
            19: 'Retired', 20: 'Homemaker', 21: 'Unemployed', 22: 'Other'
        },
        # Feature 8: Father's occupation
        8: {0: 'Student', 1: 'Rural Producer', 2: 'Specialist in Data Processing', 3: 'Specialist in Information Technologies',
            4: 'Management and Administration Technician', 5: 'Hotel, Catering, Trade and Other Services Manager',
            6: 'Accountant', 7: 'Consultant', 8: 'Construction Manager', 9: 'Commercial Manager',
            10: 'Real Estate Agent', 11: 'Lawyer', 12: 'Doctor', 13: 'Nurse', 14: 'Pharmacist',
            15: 'Teacher', 16: 'Bank Employee', 17: 'Public Administration Employee', 18: 'Other Administrative Staff',
            19: 'Retired', 20: 'Homemaker', 21: 'Unemployed', 22: 'Other'
        },
        # Feature 10: Displaced
        10: {0: 'No', 1: 'Yes'},
        # Feature 11: Educational special needs
        11: {0: 'No', 1: 'Yes'},
        # Feature 12: Debtor
        12: {0: 'No', 1: 'Yes'},
        # Feature 13: Tuition fees up to date
        13: {0: 'No', 1: 'Yes'},
        # Feature 14: Scholarship holder
        14: {0: 'No', 1: 'Yes'},
        # Feature 15: Gender (0=Male, 1=Female - VERIFY THIS INDEX from docs!)
        15: {0: 'Male', 1: 'Female'},
        # Feature 17: International student (0=No, 1=Yes - VERIFY THIS INDEX from docs!)
        17: {0: 'No', 1: 'Yes'},
        # Feature 33: Daytime/evening attendance (EXAMPLE - VERIFY from docs!)
        33: {0: 'Daytime', 1: 'Evening'},
        # Add ALL remaining CATEGORICAL features with their mappings here.
        # REFER TO UCI DATASET DOCUMENTATION (ID 697) FOR ACCURATE INDICES AND VALUES.
    }

    original_dtype_str = original_X_dtypes_map.get(feature_idx)

    if feature_idx in mappings: # Check if this numerical index has a defined mapping
        options = mappings[feature_idx]
        sorted_keys = sorted(options.keys())
        default_key = sorted_keys[0] if sorted_keys else None

        index_to_select = sorted_keys.index(default_key) if default_key is not None and default_key in sorted_keys else 0

        return st.selectbox(
            f"Select {feature_display_name} ({feature_idx}):",
            options=sorted_keys,
            format_func=lambda x: options.get(x, f"Unknown ({x})"),
            index=index_to_select,
            key=f"feature_{feature_idx}" # Ensures unique key
        )
    elif original_dtype_str == 'object': # This path is highly unlikely now with numeric column names
        return st.text_input(f"Enter {feature_display_name} ({feature_idx}):", key=f"feature_{feature_idx}")
    else: # Numerical input (int or float)
        # Use feature_display_name (string) for conditional checks.
        # You can add more specific value hints/defaults based on feature_display_name
        if 'grade' in feature_display_name.lower() or 'admission' in feature_display_name.lower() or \
           'score' in feature_display_name.lower() or 'approved' in feature_display_name.lower() or \
           'credited' in feature_display_name.lower() or 'enrolled' in feature_display_name.lower() or \
           'evaluations' in feature_display_name.lower():
            return st.number_input(f"Enter {feature_display_name} ({feature_idx}):", value=0.0, format="%.2f", key=f"feature_{feature_idx}")
        elif 'age' in feature_display_name.lower():
            return st.number_input(f"Enter {feature_display_name} ({feature_idx}):", value=18, min_value=15, max_value=80, step=1, key=f"feature_{feature_idx}")
        elif 'rate' in feature_display_name.lower() or 'gdp' in feature_display_name.lower():
            return st.number_input(f"Enter {feature_display_name} ({feature_idx}):", value=0.0, format="%.2f", key=f"feature_{feature_idx}")
        else: # Generic numerical input for any other numeric feature
            return st.number_input(f"Enter {feature_display_name} ({feature_idx}):", value=0.0, key=f"feature_{feature_idx}")


# --- Streamlit UI ---
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

st.title("🎓 Student Dropout and Academic Success Predictor")
st.markdown("""
    This application predicts whether a student is likely to **dropout**, **enroll**, or **graduate**,
    based on various academic and socio-economic factors.
""")

st.header("Student Information")

user_input = {}
cols = st.columns(3) # Use 3 columns for better layout

input_idx = 0
for feature_idx in feature_order: # Iterate through numerical indices (0-35)
    col_idx = input_idx % 3
    with cols[col_idx]:
        # Pass feature_idx (the numerical column name) to create_input_widget
        user_input[feature_idx] = create_input_widget(feature_idx, original_X_dtypes_map)
    input_idx += 1


if st.button("Predict Dropout Status"):
    # Perform prediction
    predicted_label, probabilities = predict_dropout(user_input)

    st.subheader("Prediction Result")
    if predicted_label == "Dropout":
        st.error(f"The student is predicted to be **{predicted_label.upper()}** 🙁")
    elif predicted_label == "Graduate":
        st.success(f"The student is predicted to **{predicted_label.upper()}**! 🎉")
    else: # Enrolled
        st.info(f"The student is predicted to be **{predicted_label.upper()}** 📈")

    st.markdown("---")
    st.subheader("Prediction Probabilities")
    labels = label_encoder.classes_
    prob_df = pd.DataFrame({
        'Outcome': labels,
        'Probability': probabilities
    }).sort_values(by='Probability', ascending=False)
    st.dataframe(prob_df, hide_index=True)

    st.markdown("---")
    st.caption("Disclaimer: This prediction is based on the trained model and provided input data. "
               "It should be used as an indicator and not as the sole basis for decisions.")