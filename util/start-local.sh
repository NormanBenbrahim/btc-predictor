#!/bin/bash

# TODO: add check for firestore client & database exist check

# make sure credentials folder exists
if [ ! -d "$HOME/.credentials" ]; then
    echo ""
    echo "Credentials folder not found, creating..."
    mkdir "$HOME/.credentials"
fi 

# make sure data folder exists & is empty when app is live
if [ ! -d "./data" ]; then
    mkdir "./data"
else 
    echo "continuing..."
fi 

# project specific environment variables
export SERVICE_ACCT="fast-api"
export PROJECT_ID="watch-btc-dev"

# ensure you work in same python version as on google cloud app engine
# comment these lines out after you run it the first time, otherwise it's annoying
# brew install pyenv
# brew install pyenv-virtualenv

if [ ! -d "$HOME/.pyenv/versions/3.8.6" ]; then
    pyenv install 3.8.6
    pyenv local 3.8.6
else 
    pyenv local 3.8.6
fi

# add specific version to the path
export PATH="~/.pyenv/versions/3.8.6/bin:${PATH}"

# if no venv folder, make sure to create it 
if [ ! -d "venv" ]; then
    echo ""
    echo "Local virtual environment not found, creating..."
    python3 -m venv venv
fi

# check for gcloud sdk
if [ ! -d "$HOME/google-cloud-sdk" ]; then 
    echo "" 
    echo "The google cloud SDK is not installed, go here"
    echo "https://cloud.google.com/sdk/docs/install"
    echo "and install it first, then re-run this file"
    exit
else
    echo ""
fi 

# check for credentials file
if [ ! -f "$HOME/.credentials/$PROJECT_ID.json" ]; then 
    # echo ""
    # echo "Authenticating GCP on the browser"
    # gcloud auth login

    echo ""
    echo "Setting project on GCP to $PROJECT_ID..."
    gcloud config set project "$PROJECT_ID"

    echo ""
    echo "Creating service account and setting GCP environment variable..."
    gcloud iam service-accounts create "$SERVICE_ACCT" --display-name "Service account for api created with fastAPI"
    gcloud iam service-accounts keys create "$PROJECT_ID.json" --iam-account="$SERVICE_ACCT@$PROJECT_ID.iam.gserviceaccount.com"
    mv "$PROJECT_ID.json" "$HOME/.credentials/$PROJECT_ID.json"
    export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.credentials/$PROJECT_ID.json"
else
    export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.credentials/$PROJECT_ID.json"
fi

# run app inside venv
stat venv || python3 -m venv venv
source venv/bin/activate
#pip install --use-feature=2020-resolver -r requirements.txt 
echo "source venv/bin/activate"


echo ""
echo "training model..."
start_time=`date +%s`
python predictLSTM.py && echo "run time is $(expr `date +%s` - $start_time) s"
