import pandas as pd
import numpy as np

# ჩატვირთვა
df = pd.read_csv('data/train_data.csv')

# თუ მხოლოდ ერთი კლასია
if df['defaulted'].nunique() == 1:
    # დაამატეთ რამდენიმე შემთხვევითი დეფოლტი
    new_rows = {
        'age': [45, 38],
        'income': [40000, 35000],
        'loan_amount': [20000, 15000],
        'credit_score': [600, 580],
        'years_employed': [1, 0],
        'num_credit_lines': [6, 7],
        'debt_ratio': [0.8, 0.9],
        'defaulted': [1, 1]
    }
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# შენახვა
df.to_csv('data/train_data.csv', index=False)
print("✅ შემთხვევითი დეფოლტი დამატებულია.")