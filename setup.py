
'''setup.py file: needed for project packaging in python'''

from setuptools  import find_packages, setup

def get_requirements():

    '''
    returns a list of requirements from txt file

    '''

    requirements_lst = []

    try: 
        with open("requirements.txt",'r') as file:
            lines = file.readlines()

            for line in lines:
             requirement = line.strip()
             if requirement and requirement!='-e .':
                requirements_lst.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file is not found")

    
    return requirements_lst

setup(
    name="Network_Security",
    version="0.0.1",
    author= "Anuja Parab",
    author_email= "anuja.parab4@gmail.com",
    install_requires = get_requirements(),
    packages= find_packages()

)