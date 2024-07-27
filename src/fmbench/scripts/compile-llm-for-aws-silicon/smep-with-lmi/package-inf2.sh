mkdir mymodel
cp serving-inf2.properties  mymodel/serving.properties
tar czvf mymodel-inf2.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-inf2.tar.gz s3://llm-models/lmi/Meta-Llama-3-8B-Instruct/code/
