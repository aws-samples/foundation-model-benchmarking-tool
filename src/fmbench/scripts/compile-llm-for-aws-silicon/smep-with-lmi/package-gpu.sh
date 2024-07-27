mkdir mymodel
cp serving-gpu.properties  mymodel/serving.properties
tar czvf mymodel-gpu.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-gpu.tar.gz s3://sagemaker-us-east-1-102048127330/lmi/Meta-Llama-3-8B-Instruct/code/