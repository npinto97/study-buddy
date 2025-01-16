from setuptools import setup, find_packages

setup(
    name="study-buddy",
    version="0.1",
    packages=find_packages(include=["study_buddy", "study_buddy.*"]),
)
