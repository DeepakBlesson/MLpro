from setuptools import find_packages,setup
from typing import List

HYPHEN_E_ = '-e .'
def get_requirements(file_path:str) -> list[str]:
    """
    This function will list the requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',' ')for req in requirements]
        if(HYPHEN_E_ in requirements):
            requirements.remove(HYPHEN_E_)
    requirements

setup(   
    name='MLproject',
    version='0.01',
    author='Blesson',
    author_email='deepakblessonrose14@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')

    
    )