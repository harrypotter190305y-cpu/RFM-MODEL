from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Return the list of requirements from the given file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="rfm_model",  # ✅ no spaces; safe Python package name
    version="0.0.1",
    author="Shiva",
    author_email="shivasaikuncharapu@gmail.com",
    packages=find_packages(where="src"),  # ✅ look inside the src folder
    package_dir={"": "src"},               # ✅ tell setuptools your packages live in src
    install_requires=get_requirements("requirements.txt"),
)
