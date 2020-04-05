from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


def test_read_main():
    with client:
        response = client.post("/monitor_data",
        json={
            'monitorId': 'sdfsdfsdfsd',
            'file': 'byte64 encoded file data'
            
        })
        assert response.status_code == 200
        #assert response.json() == {"Hello": "world"}

        response = client.get("/monitor_data/sdfsdfsdfsd")
        assert response.status_code == 200
        assert response.text == '{"monitorId":"sdfsdfsdfsd","imageId":0,"file":"byte64 encoded file data","timestamp":null,"ocrResults":[]}'
