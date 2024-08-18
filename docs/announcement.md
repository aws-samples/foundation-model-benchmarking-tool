# Release 2.0 announcement

We are excited to share news about a major FMBench release, we now have release 2.0 for FMBench that supports model evaluations through a panel of LLM evaluators🎉. With the recent feature additions to FMBench we are already seeing increased interest from customers and hope to reach even more customers and have an even greater impact. Check out all the latest and greatest features from FMBench on the FMBench website.


**Support for Model Evaluations**: FMBench now adds support for evaluating candidate models using Majority Voting with a [Panel of LLM Evaluators](https://arxiv.org/abs/2404.18796). Customers can now use FMBench to evaluate model accuracy across open-source and custom datasets, thus FMBench now enables customers to not only measure performance (inference latency, cost, throughput) but also model accuracy.


**Native support for LLM compilation and deployment on AWS Silicon**: FMBench now supports end-to-end compilation and model deployment on AWS Silicon. Customers no longer have to wait for models to be available for AWS Chips via SageMaker JumpStart and neither do they have to go through the process of compiling the model to Neuron themselves, FMBench does it all for them. We can simply put the relevant configuration options in the FMBench config file and it will compile and deploy the model on SageMaker ([config](https://aws-samples.github.io/foundation-model-benchmarking-tool/configs/llama3.1/8b/config-ec2-llama3-1-8b-inf2.yml)) or EC2 ([config](https://aws-samples.github.io/foundation-model-benchmarking-tool/configs/llama3.1/8b/config-ec2-llama3-1-8b-inf2-48xl-deploy-ec2.yml)).


**Website for better user experience**: FMBench has a [website](https://aws-samples.github.io/foundation-model-benchmarking-tool/) now along with an [introduction video](https://youtu.be/yvRCyS0J90c). The website is fully searchable to ease common tasks such as installation, finding the right config file, benchmarking on various hosting platforms (EC2, EKS, Bedrock, Neuron, Docker), model evaluation, etc. This website was created based on feedback from several internal teams and external customers.


**Native support for all AWS generative AI services**: FMBench now benchmarks and evaluates any Foundation Model (FM) deployed on any AWS Generative AI service, be it Amazon SageMaker, Amazon Bedrock, Amazon EKS, or Amazon EC2. We initially built FMBench for SageMaker, and later extended it to Bedrock and then based on customer requests extended it to support models on EKS and EC2 as well. See [list of config files](https://aws-samples.github.io/foundation-model-benchmarking-tool/manifest.html) supported out of the box, you can use these config files either as is or as templates for creating your own custom config.
