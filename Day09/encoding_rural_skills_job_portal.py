import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample data
df = pd.DataFrame({
    'Skill': ['Teaching', 'ಹಣೆಗಾರಿಕೆ', 'தையல் வேலை', 'Teaching'],
    'Level': ['Intermediate', 'Beginner', 'Expert', 'Expert']
})

# One-Hot Encoding for Skill
df_encoded = pd.get_dummies(df['Skill'], prefix='Skill')

# Label Encoding for Level
le = LabelEncoder()
df['Level_encoded'] = le.fit_transform(df['Level'])

# Combine
final_df = pd.concat([df, df_encoded], axis=1)
print(final_df)
