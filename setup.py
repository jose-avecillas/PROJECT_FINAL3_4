from setuptools import setup, find_packages
setup(
    name="proyecto-ia-chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy","pandas","scikit-learn","matplotlib","plotly",
        "fastapi","uvicorn","pydantic","joblib",
        "transformers","datasets","evaluate","torch","accelerate","sentencepiece"
    ],
)
