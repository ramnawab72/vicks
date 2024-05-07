
'''
#T-test

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate two samples for demonstration purposes 
np.random.seed(42)
sample1 = np.random.normal(loc=-10, scale=2, size=38)
sample2 = np.random.normal(loc=12, scale=2, size=30)

# Perform a two-sample t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)

# Set the significance level
alpha = 0.85

print("Results of Two-Sample t-test:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {len(sample1) + len(sample2) - 2}")

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.hist(sample1, alpha=0.85, label='Sample 1', color="blue")
plt.hist(sample2, alpha=0.85, label='Sample 2', color="orange")
plt.axvline(np.mean(sample1), color="blue", linestyle="dashed", linewidth=2)
plt.axvline(np.mean(sample2), color="orange", linestyle="dashed", linewidth=2)
plt.title('Distributions of Sample 1 and Sample 2')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

# Highlight the critical region if null hypothesis is rejected
if p_value < alpha:
    critical_region = np.linspace(min(sample1.min(), sample2.min()), max(sample1.max(), sample2.max()), 1000)
    plt.fill_between(critical_region, 0, 10, color='red', alpha=0.3, label='Critical Region')
    plt.text(11, 5, f'T-statistic: {t_statistic:.2f}', ha='center', va='center', color="black", backgroundcolor="white")

# Draw Conclusions
if p_value < alpha:
    if np.mean(sample1) > np.mean(sample2):
        print("Conclusion: There is significant evidence to reject the null hypothesis.")
        print("Interpretation: The mean of Sample 1 is significantly higher than that of Sample 2.")
    else:
        print("Conclusion: There is significant evidence to reject the null hypothesis.")
        print("Interpretation: The mean of Sample 2 is significantly higher than that of Sample 1.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("Interpretation: There is not enough evidence to claim a significant difference between the means.")

plt.show()

'''

#Chi-test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


df = sb.load_dataset('mpg')


print(df)


print(df['horsepower'].describe())
print(df['model_year'].describe())


bins = [0, 75, 150, 240]
df['horsepower_new'] = pd.cut(df['horsepower'], bins=bins, labels=['l', 'm', 'h'])

ybins = [69, 72, 74, 84]
labels = ['t1', 't2', 't3']
df['modelyear_new'] = pd.cut(df['model_year'], bins=ybins, labels=labels)


print(df['horsepower_new'])
print(df['modelyear_new'])


df_chi = pd.crosstab(df['horsepower_new'], df['modelyear_new'])


chi2, p, dof, expected = stats.chi2_contingency(df_chi)

print(df_chi)
print("Chi-square statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)
