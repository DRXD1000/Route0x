from setuptools import setup, find_packages
import glob
import os


build_requires = [
    'pandas==2.2.3',
    'datasets==3.2.0',
    'sentence-transformers==4.0.1',
    'setfit==1.0.2',
    'onnx==1.14.0',
    'onnxruntime==1.15.1',
    'tqdm==4.67.1',
    'scikit-learn==1.6.1',
    'huggingface_hub==0.23.5',
    'accelerate==1.5.2',
    'ollama==0.4.7',
    'matplotlib==3.10.1',
    'chardet==5.2.0',
    'openai==1.57.2',
    'anthropic==0.49.0',
    'onnxconverter-common==1.14.0',
    'faiss-cpu==1.10.0',
    'nlpaug==1.1.11',
    'kneed==0.8.5',
    'numpy==1.24.4'
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
    version='0.1.15',
    description='Low latency, High Accuracy, Custom Query routers.',
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
