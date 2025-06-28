from app import app

def test_home_route():
    test_client = app.test_client()
    response = test_client.get('/')
    assert response.status_code == 200
    assert "Ứng dụng Dự đoán Nấm" in response.data.decode('utf-8')
