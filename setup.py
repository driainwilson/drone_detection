from setuptools import setup, find_packages


def read_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name='drone_detection',
    version='0.1.0',
    description='A drone detection project',
    author='Iain Wilson',
    author_email='iainrwilson@gmail.com',
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),

    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Replace with your project's license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
