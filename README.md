# Text Taxonomy

### Problem Statement

We process documents related to mortgages, aka everything that happens to originate a mortgage that you don't see as a borrower. Often times the only access to a document we have is a scan of a fax of a print out of the document. Our system is able to read and comprehend that document, turning a PDF into structured business content that our customers can act on.

This dataset represents the output of the OCR stage of our data pipeline. Since these documents are sensitive financial documents we have not provided you with the raw text that was extracted. Instead we have had to obscure the data. Each word in the source is mapped to one unique value in the output. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from our OCR layer, so it should be _roughly_ in order.

Here is a sample line:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

The first field is the document label. Everything after the comma is a space delimited set of word values.

The dataset is included as part of this repo.

### Current State ML

The current state for the Machine Learning model deployed in this application is a static Recurrent Neural Network model. The model bears the following features:

- Built using Keras
- Utilizes a Vocabulary Processor
- Initial input layer is a word embedding layer
- Built with GRU (gated recurrent unit) cells
- About 70% test accuracy

### Current State AWS

The current state for the AWS infrastucture uses a CloudFormation template to initialize an API within an EC2 instance. The API uses Flask as the core and gunicorn as the server. The reasoning behind using EC2 instead of Lambda was due to restrictions with file sizes and packages in Lambda. The tensorflow library is too large to be supported within Lambda, and refactoring to solve this issue was outside of the targeted scope of this project. So instead the EC2 instance pulls all necessary files from S3 in the UserData section while running the service.

### Current State Automation

The project uses Jenkins as the build and deploy tool. A Jenkinsfile is included in the project which includes two stages: a stage for pushing to S3 and a stage for running the CloudFormation template. In order to utilize this Jenkins process, the tool must already be configured with your AWS credentials.

### Current State Other

This application has two endpoints: "/" and "/predictions". The "/" endpoint returns a simple html page that gives some overview of the application and also includes the ability to run a document through the "/prediction" endpoint. 

### Future State ML

- Run additional analysis of data
- Perform additional feature engineering on dataset (undersample to fix class imbalance for starters)
- Implement automated hyperparameter tuning
- Try other types of models
- Utilize more automated framework such as SageMaker (non-free tier!)

### Future State AWS

- Switch to Lambda (solve package size limit issues)
- Use SageMaker for training and deployment (non-free tier!)
- Add more automation into the CFT

### Future State Automation

- Add stage for hyperparameter tuning
- Add stage for creating the model

### Future State Other

- Add ability to write incoming documents and their associated model prediction values to a location (probably an S3 bucket) for further training / improving the model