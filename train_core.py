import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_and_prepare_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    df = pd.read_csv(url, header=None, names=columns)

    replace_map = {}
    if df['stalk-root'].isin(['?']).any():
        mode = df['stalk-root'][df['stalk-root'] != '?'].mode()[0]
        df['stalk-root'] = df['stalk-root'].replace('?', mode)
        replace_map['stalk-root'] = mode

    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders, replace_map

# Huấn luyện để lấy số mẫu → cập nhật giá trị k cho KNN
df_for_k, _, _ = load_and_prepare_data()
X_for_k = df_for_k.drop('class', axis=1)
k = int(np.sqrt(len(X_for_k)))

models = {
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=k)
}
if __name__ == "__main__":
    df, encoders, _ = load_and_prepare_data()
    print("Số mẫu:", len(df))
    print("KNN đang dùng k =", models["KNN"].n_neighbors)
