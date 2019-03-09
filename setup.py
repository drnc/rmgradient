from setuptools import setup

setup(
    name='rmgradient',
    version='1.0',
    description='Substract a background gradient from a TIFF file image',
    url='https://github.com/drnc/rmgradient',
    author='Guillaume Duranceau',
    author_email='g2lm@drnc.net',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    keywords='remove background gradient TIFF',
    py_modules=['rmgradient/rmgradient'],
    install_requires=[
        'numpy>=1.14', 'tifffile>=2018.11.6', 'imagecodecs>=2018.11.6',
        'scipy>=1.0'],
    python_requires='>=3',
    entry_points={
        'console_scripts': [ 'rmgradient=rmgradient.rmgradient:main' ],
    }
)
