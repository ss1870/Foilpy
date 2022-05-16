from setuptools import setup, find_packages

setup(
    name='foilpy',
    version='0.0',
    description='Package for the analysis of hydrofoils. Uses vortex lifting line theory to compute loads.',
    project_urls={"repository": "https://github.com/ss1870/Foilpy"},
    author='Samuel Scott',
    author_email='ss1870@my.bristol.ac.uk',
    license='Apache-2.0',
    # packages=find_packages()
    packages=['foilpy', 'foilpy.myaeropy'],
    include_package_data=True
)