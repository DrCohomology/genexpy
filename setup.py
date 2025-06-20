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
    description='A package to quantify the generalizability of experimental studies.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="anonym",
    author_email='anonym@anonym.com',
    url='anonym.com',
    install_requires=read_requirements(),
    python_requires='>=3.11.8',
)
