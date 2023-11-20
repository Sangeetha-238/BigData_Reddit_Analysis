# script to build a SageMaker Processing Job Spark container with spark-nlp

# set the params used by this script, some are determined at runtime and some 
# are read from the params.txt file
export REGION=$(aws configure get region)
echo "export REGION=${REGION}"

export ACCOUNT_ID=$(aws sts get-caller-identity | jq -r '.Account')
export IMAGE_NAME=latest
export REPO_NAME=sagemaker-spark
export BASE_IMAGE=173754725891.dkr.ecr.us-east-1.amazonaws.com/sagemaker-spark-processing:3.4-cpu-py39-v1.0

echo region=$REGION, image=${IMAGE_NAME}, account=${ACCOUNT_ID}, image=${IMAGE_NAME}, repo=${REPO_NAME}, base image=${BASE_IMAGE}

#############################################################################
# Step 1. Create the container
#############################################################################

# login to ECR for the base image
aws --region ${REGION} ecr get-login-password | docker login --username AWS --password-stdin ${BASE_IMAGE}

# create a Dockerfile
echo FROM "${BASE_IMAGE}" > Dockerfile
echo RUN /usr/local/bin/pip3 install spark-nlp==5.1.3 >> Dockerfile

# Create the ECR repository if it does not already exist

# use the full word match -w switch with grep, dont want to match smstudio-custom with smstudio-custom-cv
aws --region us-east-1 ecr describe-repositories | jq .repositories[].repositoryName  | grep -w  ${REPO_NAME}
if [ $? -eq 0 ]; then
   echo "${REPO_NAME} already exists, no need to create it"
else
   echo "${REPO_NAME} does not exist, going to create it now.."
   cmd="aws --region ${REGION} ecr create-repository --repository-name ${REPO_NAME}"
   echo going to run cmd="${cmd}"
   ${cmd}
   if [ $? -eq 0 ]; then
       echo "created ECR repo ${REPO_NAME}"
    else
       echo "failed to create ECR repo ${REPO_NAME}"
       exit 1
    fi
fi

# Build the image - it might take a few minutes to complete this step
image=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_NAME}
echo building the container image ${image}
docker build . -t ${IMAGE_NAME} -t ${image}
if [ $? -eq 0 ]; then
    echo ${image} created successfully
else
    echo ${image} creation failed, exiting...
    exit 1
fi

# Login to ECR
echo logging into ECR to push image we just built
aws --region ${REGION} ecr get-login-password | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}

# Push the image to ECR
echo pushing ${image} to ECR
docker push ${image}
if [ $? -eq 0 ]; then
    echo ${image} pushed successfully
else
    echo ${image} push failed, exiting...
    exit 1
fi

echo "all done"

