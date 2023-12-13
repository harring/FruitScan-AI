# FruitScan CNN Project

## Overview
FruitScan is a web-application that predicts which fruit is in an image and provides the user with nutritional information about that fruit. This allows the user to learn about fruits and make educated choices when eating healthy.
This project uses a Convolutional Neural Network (CNN) implemented in Django. It currently supports identification of kiwi, mango, banana, and tomato, with the capability to extend to more fruits. 

## Features
- CNN for fruit prediction.
- Docker and Docker-Compose integration.
- GitHub Actions for continuous integration.
- Unit tests to ensure code reliability.
- Kubernetes deployment configurations.

## Project Status

This project was developed as a part of the course "DIT826 H23 Software Engineering for Data-Intensive AI Applications" at the University of Gothenburg. As the course has concluded, the development of this project has also reached its completion. Therefore, the project is currently in a maintenance phase and will not be actively developed further. However, the codebase remains available for educational purposes, future reference, and community use.

## Screenshots

Below are some screenshots illustrating key features of our application:

### FruitScan Logo
<img src="https://github.com/harring/fruitscan-1/blob/main/FruitScan/fruitscanapp/static/images/FruitScan_logo.png?raw=true" width="300">


### Administrator Menu
<img src="https://github.com/harring/fruitscan-1/blob/main/FruitScan/fruitscanapp/static/images/Banana.jpg" width="300">


*This screenshot displays the administrator menu, showcasing the backend control features available to administrators. Administrators can add/remove images to the training and test sets. They can also evaluate and deploy different model versions.*

### Fruit Prediction Page
<img src="https://github.com/harring/fruitscan-1/blob/main/FruitScan/fruitscanapp/static/images/Banana.jpg" width="300">


*Here, you can see a sample page where a fruit is predicted by our CNN model.*

### Explainability Feature
<img src="https://github.com/harring/fruitscan-1/blob/main/FruitScan/fruitscanapp/static/images/Banana.jpg" width="300">


*This image illustrates the explainability aspect of our application, detailing how the CNN model arrives at its predictions.*

## CNN Design

### CNN Architecture Overview
<img src="https://github.com/harring/fruitscan-1/blob/main/FruitScan/fruitscanapp/static/images/cnn_model.png?raw=true">


*This image provides an overview of the CNN architecture used in our project. It details the various layers and their configurations within our fruit prediction model.*


## Getting Started

### Prerequisites
- Docker
- Kubernetes
- Google Cloud SDK

### Installation
1. Clone the repository:
`git clone [repository-url]`

2. Build the Docker image:
`docker-compose build`

3. Run the Docker container:
`docker-compose up`


### Connecting to Kubernetes Cluster
To connect to the Kubernetes cluster:

1. **Install kubectl:** 
- Follow the instructions at [Kubernetes kubectl](https://kubernetes.io/docs/tasks/tools/).

2. **Install Google Cloud SDK:** 
- Instructions available at [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

3. **Initialize gcloud:** 
- If not prompted to log in, run `gcloud init`.
- When prompted to pick a project, abort (Ctrl+C).

4. **Get kubectl credentials for the project(example command):**
`gcloud container clusters get-credentials sample-cluster --location=us-central1-f `

5. **Install any required plugins:** 
- If warned about a missing plugin (e.g., gcloud-auth-plugin), install it.

6. **Accessing the project:**
- Use `kubectl` to interact with the cluster:
  - `kubectl get pods` shows active pods.
  - `kubectl get nodes` shows active nodes (`-o wide` for more info).
  - `kubectl get services` shows services.

### Kubernetes Management
- **To delete pods:** (Use with caution)
`kubectl delete pods -l app=web`

- **To update deployment configuration:**
- Navigate to the `deployment.yaml` file directory.
- Apply the new configuration:
  `kubectl apply -f deployment.yaml`

## Tech Stack

Our project leverages a robust stack of technologies and libraries. Below is a breakdown of the key components:

### Core Technologies
- **Django:** A high-level Python Web framework that encourages rapid development and clean, pragmatic design.
- **TensorFlow:** An open-source software library for machine learning, used for the CNN model in our project.
- **Keras:** A deep learning API written in Python, running on top of the machine learning platform TensorFlow.
- **Docker:** A set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers.
- **Kubernetes:** An open-source platform for automating the deployment, scaling, and management of containerized applications, organizing them into logical units for efficient operation.

### Data Science and Machine Learning Libraries
- **NumPy:** A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
- **Pandas:** A software library written for data manipulation and analysis.
- **Scikit-Learn:** A free software machine learning library for Python.
- **Matplotlib** and **Seaborn:** Libraries for creating static, interactive, and animated visualizations in Python.

### Web Development Libraries
- **Flask:** A micro web framework written in Python.
- **Jinja2:** A modern and designer-friendly templating language for Python, modeled after Django’s templates.

### Other Key Libraries
- **opencv-python:** A library of Python bindings designed to solve computer vision problems.
- **Pillow:** The Python Imaging Library adds image processing capabilities to your Python interpreter.
- **Requests:** An elegant and simple HTTP library for Python.
- **h5py:** A Pythonic interface to the HDF5 binary data format.

### Development and Testing Tools
- **ipython, Jupyter:** Tools for interactive computing in Python.
- **pytest:** A framework for easily writing small tests, yet scales to support complex functional testing.

This list represents a snapshot of the primary technologies used in our project. For a full list of dependencies, please refer to our `requirements.txt` file.


## License
This project is licensed under the MIT License. 


## Developers/Collaborators

For any queries or contributions, feel free to reach out to our team:

- Erik Harring - [harring](https://github.com/harring)
- Patricia Marklund - [PatyMarklund](https://github.com/PatyMarklund)
- Mijin Kim - [mezyn](https://github.com/mezyn)
- Jonathan Bergdahl - [jonathanb00](https://github.com/jonathanb00)

