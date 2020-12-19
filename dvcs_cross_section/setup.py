from setuptools import setup, find_packages

setup(
    name="dvcs_xsx",
    version="1.0.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="Predict DVCS cross sections with Deep Learning",
    author="SIWIF",
    author_email="uvafemtography@gmail.com",
    license="MIT",
    packages=find_packages(),
)
