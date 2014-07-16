try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='maxCutPy',
    version='0.1.0',
    author=['Andrea Casini', 'Nicola Rebagliati'],
    author_email=['andreacasini88@gmail.com', 'nicola.rebagliati@gmail.com'],
    packages=['maxcutpy'],
    license='GPL',
    description='A Python library for solving the Max Cut problem',
    long_description=open('README.md').read(),
    install_requires=['numpy',
                      'networkx',
                      'scipy',
                      'matplotlib']
)

