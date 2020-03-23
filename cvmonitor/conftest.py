import pytest
from .server import Server

@pytest.fixture
def app():
    server = Server()
    return server.app
