from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_read_main():
    with client:
        data = {"imageId": "image", "deviceId": "device", "timestamp": "", "segments": [], "image": "blah"}
        response = client.post("/monitor_data", json=data)
        assert response.status_code == 200  # TODO maybe 202 ?

        response = client.get("/monitor_data/device")
        assert response.status_code == 200
        assert response.json() == {"deviceId": "device", "imageId": "image", "timestamp": "", "segments": [], "image": "blah"}
