from setuptools import setup, find_packages

setup(
    name='releasy',
    version='1.0.0',
    packages=find_packages(where='releasy'),
    package_dir={'': 'releasy'},
    install_requires=[],
    description="Releasy is a tool that collects provenance data from releases by parsing the software version control and issue tracking systems.",
    author="Felipe Curty",
    author_email="felipecrp@gmail.com",
    url="https://github.com/gems-uff/releasy", 
    python_requires='>=3.7',
)
