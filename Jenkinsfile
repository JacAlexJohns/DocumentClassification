pipeline {
    agent {
        docker {
            image 'python:3.5.1'
        }
    }
    environment {
        AWS_ID = credentials('aws-credentials')
        AWS_ACCESS_KEY_ID = "${env.AWS_ID_USR}"
        AWS_SECRET_ACCESS_KEY = "${env.AWS_ID_PSW}"
    }
    stages {
        stage('Upload to S3') {
            steps {
                sh 'pip3 install awscli'
                sh 'aws s3 cp ./src s3://test-ml-bucket/src --recursive'
            }
        }
        stage('Run CloudFormation Template') {
            steps {
                sh '''
                aws cloudformation validate-template --region us-east-1 --template-body file://CloudFormation.yaml
                aws cloudformation create-stack --stack-name ml-service --region us-east-1 --template-body file://CloudFormation.yaml --capabilities CAPABILITY_NAMED_IAM
                '''
            }
        }
    }
}