#!/usr/bin/env python3
"""
Setup script for Emotion Recognition System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements files
def read_requirements(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle platform-specific packages
                    if 'python_version' in line:
                        requirements.append(line)
                    elif not any(skip in line for skip in ['Built-in', 'uncomment', 'download']):
                        requirements.append(line.split()[0])  # Take only package name
            return requirements
    except FileNotFoundError:
        return []

# Core requirements
core_requirements = [
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'opencv-python>=4.5.0',
    'numpy>=1.21.0',
    'Pillow>=8.3.0',
    'transformers>=4.20.0',
    'scikit-learn>=1.0.0',
    'SpeechRecognition>=3.8.0',
    'pyaudio>=0.2.11',
    'tqdm>=4.64.0',
    'pandas>=1.5.0'
]

# Optional requirements for different components
extras_require = {
    'multimodal': read_requirements('requirements/requirements_multimodal.txt'),
    'furhat': read_requirements('requirements/requirements_furhat.txt'),
    'voice_ter': read_requirements('requirements/requirements_voice_ter.txt'),
    'camera_inference': read_requirements('requirements/requirements_camera_inference.txt'),
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0.0',
        'isort>=5.0.0',
        'flake8>=3.8.0',
        'mypy>=0.812'
    ]
}

# All requirements
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name='emotion-recognition-system',
    version='1.0.0',
    author='Henry Ward',
    author_email='45144290+kudosscience@users.noreply.github.com',
    description='Multimodal Emotion Recognition System with FER and TER capabilities',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/kudosscience/fer_and_ter_model',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'fer-camera=src.fer.camera_fer_inference:main',
            'ter-voice=src.ter.voice_ter_inference:main',
            'multimodal-emotion=src.multimodal.multimodal_emotion_inference:main',
            'furhat-emotion=src.furhat.furhat_multimodal_emotion_inference:main',
        ],
    },
    package_data={
        'src': [
            '../models/*.pth',
            '../models/ter_distilbert_model/*',
            '../datasets/*.json',
            '../docs/*.md',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'emotion recognition',
        'facial expression recognition',
        'text emotion recognition',
        'multimodal ai',
        'computer vision',
        'natural language processing',
        'machine learning',
        'deep learning',
        'pytorch',
        'transformers'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/kudosscience/fer_and_ter_model/issues',
        'Source': 'https://github.com/kudosscience/fer_and_ter_model',
        'Documentation': 'https://github.com/kudosscience/fer_and_ter_model/blob/main/README.md',
    },
)
