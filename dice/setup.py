from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='dice',
    version='0.1',
    description='A dice model for interactive object recognition',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
    ],
    keywords='',
    url='https://github.com/ccorcos/',
    author='Chet Corcos',
    author_email='ccorcos@gmail',
    license='MIT',
    packages=['dice'],
    install_requires=[
        'numpy',
        'se3',
        'sparkprob'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
