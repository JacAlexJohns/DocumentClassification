pipeline {
    agent {
        docker {
            image 'fstab/aws-cli'
        }
    }
    stages {
        stage('Upload to S3') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'aws-creds',
                                                  accessKeyVariable: 'AWS_ACCESS_KEY_ID',
                                                  secretKeyVariable: 'AWS_SECRET_ACCESS_KEY')]) {
                        sh 'aws s3 cp ./src s3://test-ml-bucket/src --recursive'
                    }
                }
            }
        }
        stage('Run CloudFormation Template') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'aws-creds',
                                                  accessKeyVariable: 'AWS_ACCESS_KEY_ID',
                                                  secretKeyVariable: 'AWS_SECRET_ACCESS_KEY')]) {
                        sh '''
                        aws cloudformation validate-template --region us-east-1 --template-body file://CloudFormation.yaml
                        aws cloudformation create-stack --stack-name ml-service --region us-east-1 --template-body file://CloudFormation.yaml --capabilities CAPABILITY_NAMED_IAM
                        '''
                    }
                }
            }
        }
    }
}