import pytest

@pytest.fixture(scope='session')
def app():
    from .server import Server
    server = Server()
    return server.app
