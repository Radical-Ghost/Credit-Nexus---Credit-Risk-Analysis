import pandas as pd
import numpy as np
import joblib

# Load saved scaler
scaler = joblib.load('scaler.pkl')

# Define the function
def preprocess_input(user_input):
    """
    user_input: list of values in the following order:
    [dti, annual_inc, loan_amnt, revol_util, delinq_2yrs,
    inq_last_6mths, open_acc, pub_rec, bc_util, revol_bal,
    total_acc, mo_sin_old_rev_tl_op, num_actv_bc_tl, percent_bc_gt_75,
    term, home_ownership, verification_status, purpose]
    
    term: '36 months' or '60 months'
    home_ownership: 'MORTGAGE', 'RENT', 'OWN', 'OTHER'
    verification_status: 'Not Verified', 'Verified', 'Source Verified'
    purpose: one of main purposes or 'OTHER'
    """

    # Convert to DataFrame
    df_input = pd.DataFrame([user_input], columns=[
        'dti', 'annual_inc', 'loan_amnt', 'revol_util', 'delinq_2yrs',
        'inq_last_6mths', 'open_acc', 'pub_rec', 'bc_util', 'revol_bal',
        'total_acc', 'mo_sin_old_rev_tl_op', 'num_actv_bc_tl', 'percent_bc_gt_75',
        'term', 'home_ownership', 'verification_status', 'purpose'
    ])
    
    # TERM mapping
    term_map = {'36 months': 0, '60 months': 1}
    df_input['term'] = df_input['term'].str.strip().map(term_map).astype('Int64')
    
    # HOME OWNERSHIP
    df_input['home_ownership'] = df_input['home_ownership'].str.upper().replace({'NONE':'OTHER', 'ANY':'OTHER'})
    keep = ['MORTGAGE', 'RENT', 'OWN']
    df_input['home_ownership_grp'] = df_input['home_ownership'].where(df_input['home_ownership'].isin(keep), 'OTHER')
    
    # One-hot encode home_ownership
    for col in ['MORTGAGE', 'OWN', 'RENT']:
        df_input[f'home_ownership_grp_{col}'] = (df_input['home_ownership_grp'] == col).astype(int)
    df_input.drop(['home_ownership_grp', 'home_ownership'], axis=1, inplace=True)
    
    # VERIFICATION STATUS
    verification_map = {'Not Verified': 0, 'Verified': 1, 'Source Verified': 2}
    df_input['verification_status'] = df_input['verification_status'].map(verification_map).astype('int64')
    
    # PURPOSE
    main_purposes = ['debt_consolidation', 'credit_card', 'house', 'car', 
                    'home_improvement', 'small_business', 'major_purchase']
    df_input['purpose_grp'] = df_input['purpose'].where(df_input['purpose'].isin(main_purposes), 'OTHER')
    
    # One-hot encode purpose, drop first to match training
    for col in main_purposes[1:]:  # skip first for drop_first
        df_input[f'purpose_grp_{col}'] = (df_input['purpose_grp'] == col).astype(int)
    df_input.drop(['purpose', 'purpose_grp'], axis=1, inplace=True)
    
    # NUMERIC SCALING
    numeric_cols = ['dti', 'annual_inc', 'loan_amnt', 'revol_util', 'delinq_2yrs', 
                    'inq_last_6mths', 'open_acc', 'pub_rec', 'bc_util', 'revol_bal', 
                    'total_acc', 'mo_sin_old_rev_tl_op', 'num_actv_bc_tl', 'percent_bc_gt_75']
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    return df_input



#how it will execute
user_input = [
    20.0,        # dti
    75000,       # annual_inc
    15000,       # loan_amnt
    30.0,        # revol_util
    0,           # delinq_2yrs
    1,           # inq_last_6mths
    10,          # open_acc
    0,           # pub_rec
    25.0,        # bc_util
    5000,        # revol_bal
    20,          # total_acc
    120,         # mo_sin_old_rev_tl_op
    5,           # num_actv_bc_tl
    10.0,        # percent_bc_gt_75
    '36 months', # term
    'RENT',      # home_ownership
    'Verified',  # verification_status
    'credit_card'# purpose
]

processed = preprocess_input(user_input)

# Now you can predict
# prediction = best_xgb_default_loaded.predict(processed)
# print(prediction)
