#!/bin/bash
set -e

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 439543761220.dkr.ecr.us-east-1.amazonaws.com
docker build -t alfred/ona_streamlit .
docker tag alfred/ona_streamlit:latest 439543761220.dkr.ecr.us-east-1.amazonaws.com/alfred/ona_streamlit
docker push 439543761220.dkr.ecr.us-east-1.amazonaws.com/alfred/ona_streamlit:latest