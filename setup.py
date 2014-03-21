from setuptools import setup, find_packages


setup(
    name = 'SDdata',
    version = 0.1,

    packages = find_packages(),
    scripts=[],
    requires = ['numpy', 'h5py'],

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui",
    author_email = "kiyo@physics.ubc.ca",
    description = "Single Dish Data",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/SDdata"
)
