from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["vocabulary_size"] == 94


def test_predict_spam():
    res = client.post(
        "/api/predict",
        json={"message": "Congratulations! You have won a free cruise trip."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Spam"
    assert data["is_spam"] is True
    assert data["spam_probability"] > 0.5


def test_predict_ham():
    res = client.post(
        "/api/predict",
        json={"message": "Hey, don't forget our dinner plan at 8."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Ham"
    assert data["is_spam"] is False
    assert data["ham_probability"] > 0.5


def test_ui_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Spam Email Classifier" in res.text


if __name__ == "__main__":
    test_health()
    test_predict_spam()
    test_predict_ham()
    test_ui_served()
    print("All API and UI tests passed!")
