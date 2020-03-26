from setuptools_scm import get_version
try:
    
    __version__ = get_version()
except PackageNotFoundError:
    # package is not installed
   pass