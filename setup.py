import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phypro-pkg-kerzip", # Replace with your own username
    version="0.0.1",
    author="Kerstin Pieper",
    author_email="author@example.com",
    description="A small physio processing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerzip/physio-processing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['wheel'],
)