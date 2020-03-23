from setuptools import setup, find_packages


setup(
    name="monitors",
    packages=find_packages("monitors"),
    package_dir={"": "monitors"},
    install_requires=open("requirements.txt").readlines(),
    entry_points = {
        'console_scripts': ['start-server=monitors.server:main'],
    }
)