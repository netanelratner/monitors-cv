from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_read_main():
    with client:
        monitorId = "monitor"
        imageId = "image"
        timestamp = "2020-04-05 19:48:45.562424"
        data = {"imageId": imageId, "monitorId": monitorId, "timestamp": timestamp, "segments": []}
        response = client.post("/monitor_data/" + monitorId, json=data)
        assert response.status_code == 200  # TODO maybe 202 ?

        data = b"an base64 encoded image"
        response = client.post("/monitor_image/" + monitorId, data=data, headers={
            "X-IMAGE-ID": imageId,
            "X-MONITOR-ID": monitorId,
            "X-TIMESTAMP": timestamp
        })
        assert response.status_code == 200  # TODO maybe 202 ?



        #response = client.get("/monitor_data/monitor")
        #assert response.status_code == 200
        #assert response.json() == {"deviceId": "device", "imageId": "image", "timestamp": "", "segments": [], "image": "blah", "deviceCategory": "category"}
