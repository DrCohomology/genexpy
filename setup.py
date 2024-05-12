from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements

setup(
    name='genexpy',
    version='0.1.0',
    packages=find_packages(),
    description='A package to quantify the generalizability of experimental results',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Federico Mateucci, Konstantin Ntounas',
    author_email='federico.matteucci@kit.edu, uypgi@student.kit.edu',
    url='https://github.com/DrCohomology/Generalizability-of-experimental-comparisons',
    install_requires=read_requirements(),
    python_requires='>=3.12.3',
)
