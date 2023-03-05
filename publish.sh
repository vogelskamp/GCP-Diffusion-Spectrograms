rm -f ./credentials.json

gcloud auth application-default login

cp /Users/samvogelskamp/.config/gcloud/application_default_credentials.json .

mv ./application_default_credentials.json ./credentials.json

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=diffusion-project-repository
export IMAGE_NAME=cuda-diffusion
export IMAGE_TAG=0.1
export IMAGE_URI=europe-west1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -f Dockerfile -t ${IMAGE_URI} ./

docker push ${IMAGE_URI}

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=diffusion-spectrograms \
  --worker-pool-spec=machine-type=a2-highgpu-4g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=4,replica-count=1,container-image-uri=europe-west1-docker.pkg.dev/dev-diffusion-project/diffusion-project-repository/cuda-diffusion:0.1