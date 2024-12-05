FastAPI Project with Docker
This application generates BMC for a user idea input in multipe Language. you can try it by acceding the swagger documentation on this url: http://127.0.0.1:8000/docs

Project Structure
docker-compose.yml: Defines the services and configurations needed to run the application in a Docker container.
Dockerfile: A Docker configuration file to build and run the application in a container.
grouped_df.pkl: A large dataset file in pickle format that contains our Preprocessed ESCO Dataset.
main.py: The main Python script that defines the FastAPI application and its endpoints.
requirements.txt: A list of dependencies required for the project.
Requirements
Python 3.10+
Docker (if you want to run the application in a container)
Git LFS (for handling large files, specifically grouped_df.pkl)
Installation
1. Clone the Repository
To get started, clone the repository to your local machine:

Installation
git clone --branch master --single-branch git@bitbucket.org:smrtbio/bmc.git cd bmc

1/ if running locally :
1. Install Dependencies
If you want to run the FastAPI application locally without Docker, install the required dependencies:

1.1. you can make a virtual environment
creating the environement
python -m venv env

Activate the env
.\env\Scripts\activate

1.2. installing the requirements
pip install -r requirements.txt

2. Ensure Git LFS is Installed
if it is not pulled , Since grouped_df.pkl is a large file, it�s stored using Git LFS. Ensure Git LFS is installed and pull the large files:

git lfs install git lfs pull

3. Running the Application Locally
You can run the FastAPI application locally using uvicorn:

uvicorn main:app --reload ( This will start the server at http://127.0.0.1:8000. )

2/Running with Docker (you should have docker desktop running if you are on windows , if you are on linux you are good ) :
To run the application in a Docker container:

commands To Run :
git clone --branch master --single-branch git@bitbucket.org:smrtbio/bmc.git

cd bmc

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs

git lfs install

git lfs pull

1. Start the service with Docker Compose:
First, build the Docker image using the following command:

docker-compose up

2. To stop the service:
docker-compose down
