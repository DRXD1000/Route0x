from setuptools import setup, find_packages
import glob
import os


build_requires = [
    'pandas',
    'datasets',
    'sentence-transformers',
    'setfit==1.0.2',
    'onnx==1.14.0',
    'onnxruntime==1.15.1',
    'tqdm',
    'scikit-learn',
    'huggingface_hub==0.23.5',
    'accelerate',
    'ollama',
    'matplotlib',
    'chardet',
    'openai==1.37.1',
    'anthropic',
    'onnxconverter-common',
    'faiss-cpu',
    'nlpaug'
]

route_requires = [
    'numpy==1.24.4',
    'tokenizers==0.19.1',
    'onnxruntime==1.15.1',
    'joblib==1.4.2',
    'faiss-cpu',
    'seaborn',
]


setup(
    name='Route0x',
    version='0.0.41',
    description='Low latency, High Accuracy, Custom Query routers for Humans and Agents',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[],
    extras_require={
        'build': build_requires,
        'route': route_requires,
    },
    package_data={
        'route0x.route_builder': ['*.json', 'data/*'] 
    },
    include_package_data=True, 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  
)
