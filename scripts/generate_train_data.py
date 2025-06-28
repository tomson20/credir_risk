import pandas as pd
import numpy as np

np.random.seed(42)

# 5000 სიმულაციური მონაცემი
n_samples = 5000

data = {
    'age': np.random.randint(18, 70, n_samples),
    'income': np.round(np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)),
    'loan_amount': np.round(np.random.uniform(5000, 50000, n_samples), 2),
    'credit_score': np.random.randint(300, 850, n_samples),
    'years_employed': np.random.randint(0, 30, n_samples),
    'num_credit_lines': np.random.poisson(lam=3, size=n_samples),
    'debt_ratio': np.round(np.random.uniform(0, 1, n_samples), 2),
    'defaulted': np.random.binomial(n=1, p=0.1, size=n_samples)  # 10% დეფოლტი
}

df = pd.DataFrame(data)
df.to_csv('data/train_data.csv', index=False)
print("✅ train_data.csv წარმატებით შეიქმნა.")