from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_read_main():
    with client:
        data = {"imageId": "image", "monitorId": "monitor", "timestamp": "2020-04-05 19:48:45.562424", "segments": []}
        response = client.post("/monitor_data/" + data["monitorId"], json=data)
        assert response.status_code == 200  # TODO maybe 202 ?

        #response = client.get("/monitor_data/monitor")
        #assert response.status_code == 200
        #assert response.json() == {"deviceId": "device", "imageId": "image", "timestamp": "", "segments": [], "image": "blah", "deviceCategory": "category"}
