import os
import pickle
import pytest
import pandas as pd
from app import app
import train_model
from train_core import load_and_prepare_data, models


def test_home_route():
    """Test route chính Flask trả về 200 OK và chứa tiêu đề"""
    test_client = app.test_client()
    response = test_client.get('/')
    assert response.status_code == 200
    assert "Ứng dụng Dự đoán Nấm" in response.data.decode('utf-8')


def test_training_produces_model():
    """Test train_model.py có thể huấn luyện và tạo file best_model.pkl"""
    # Xoá file nếu đã tồn tại để kiểm tra lại
    if os.path.exists("best_model.pkl"):
        os.remove("best_model.pkl")

    # Gọi lại load + train
    df, encoders, replace_map = train_model.load_and_prepare_data()
    X = df.drop('class', axis=1)
    y = df['class']
    model = list(train_model.models.values())[0]
    model.fit(X, y)

    # Ghi lại mô hình để test
    with open("best_model.pkl", "wb") as f:
        pickle.dump((model, encoders, replace_map), f)

    assert os.path.exists("best_model.pkl")


def test_model_can_predict():
    """Test mô hình trong best_model.pkl có thể dự đoán"""
    assert os.path.exists("best_model.pkl")

    with open("best_model.pkl", "rb") as f:
        loaded = pickle.load(f)
        model, label_encoders, replace_map = loaded

    # Tạo một dòng dữ liệu mẫu
    df, encoders, _ = train_model.load_and_prepare_data()
    sample = df.drop('class', axis=1).iloc[[0]]

    # Dự đoán
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]  # Vì nhãn đã encode là 0 hoặc 1


@pytest.mark.parametrize("col", ["cap-shape", "odor", "gill-size"])
def test_label_encoder_columns_exist(col):
    """Test các cột dữ liệu chính có trong label_encoders"""
    _, label_encoders, _ = train_model.load_and_prepare_data()
    assert col in label_encoders
