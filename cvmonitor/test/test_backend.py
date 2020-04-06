import pytest
import os
import random
import datetime
import time
from fastapi.testclient import TestClient

from ..backend import main
from ..generator import generate

@pytest.fixture(scope="module")
def client():

    name = "".join([random.choice([c for c in "abcdefghijklmnop"]) for i in range(30)])
    port = random.randint(6000, 7000)
    cmd = f"docker run --net host --rm -d  --name {name} redis --port {port}"
    print(cmd)
    os.system(cmd)
    os.environ["CVMONITOR_BACKEND_REDIS_PORT"] = str(port)
    server = main.Server()
    with TestClient(server.app) as client:
        yield client
    #os.system(f"docker stop {name}")
    os.system(f"docker rm -f {name}")



def test_recording_flow2(client):
    gdevice = generate.Device('monitor', 'p123', 100)
    res = generate.send_picture('',gdevice, client.post)
    assert int(res['nextImage'])==1

def test_recording_flow3(client):
    gdevice = generate.Device('monitor2', 'p1234', 101)
    res = generate.send_picture('',gdevice, client.post)
    monitor_data = client.get(f'/monitor_data/{gdevice.monitor_id}')
    import pdb; pdb.set_trace()
    assert int(res['nextImage'])==1
    res = generate.add_device('',gdevice, client.post)
    res = generate.send_picture('',gdevice, client.post)
    assert int(res['nextImage'])==2
    
def test_recording_flow(client):
    monitorId = "monitor"
    imageId = "1237"
    timestamp = "2020-04-05 19:48:45.562424"
    data = {
        "imageId": imageId,
        "monitorId": monitorId,
        "timestamp": timestamp,
        "segments": [],
    }
    response = client.post("/monitor_data/" + monitorId, json=data)
    assert response.status_code == 200  # TODO maybe 202 ?
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    response = client.post(
        "/monitor_image/" + monitorId,
        data= image,
        headers={
            "X-IMAGE-ID": imageId,
            "X-MONITOR-ID": monitorId,
            "X-TIMESTAMP": timestamp,
        },
    )
    assert response.status_code == 200  # TODO maybe 202 ?


def test_setup_flow(client):
    with client:
        monitorId = "monitor"
        imageId = "1237"
        timestamp = datetime.datetime.now().isoformat('T')
        data = {
            "imageId": imageId,
            "monitorId": monitorId,
            "timestamp": timestamp,
            "segments": [],
        }
        response = client.post("/monitor_data/" + monitorId, json=data)
        assert response.status_code == 200  # TODO maybe 202 ?
        image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
        response = client.post(
            "/monitor_image/" + monitorId,
            data= image,
            headers={
                "X-IMAGE-ID": imageId,
                "X-MONITOR-ID": monitorId,
                "X-TIMESTAMP": timestamp,
            },
        )
        assert response.status_code == 200  # TODO maybe 202 ?
