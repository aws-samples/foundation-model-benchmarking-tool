# Running multiple model copies on Amazon EC2

It is possible to run multiple copies of a model if the tensor parallelism degree and the number of GPUs/Neuron cores on the instance allow it. For example if a model can fit into 2 GPU devices and there are 8 devices available then we could run 4 copies of the model on that instance. Some inference containers, such as the [DJL Serving LMI](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) automatically start multiple copies of the model within the same inference container for the scenario described in the example above. However, it is also possible to do this ourselves by running multiple containers and a load balancer through a Docker compose file. `FMBench` now supports this functionality by adding a single parameter called `model_copies` in the configuration file.

For example, here is a snippet from the [config-ec2-llama3-1-8b-p4-tp-2-mc-max](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/1db3cdd09ba4dafc095f3c5313fcd5dd1a48313c/src/fmbench/configs/llama3.1/8b/config-llama3.1-8b-trn1-32xl-deploy-tp-8-ec2.yml#L199) config file. The new parameters are `model_copies`, `tp_degree` and `shm_size` in the `inference_spec` section. **_Note that the `tp_degree` in the `inference_spec` and `option.tensor_parallel_degree` in the `serving.properties` section should be set to the same value_**.

```{.bash}
    inference_spec:
      # this should match one of the sections in the inference_parameters section above
      parameter_set: ec2_djl
      # how many copies of the model, "1", "2",..max
      # set to 1 in the code if not configured,
      # max: FMBench figures out the max number of model containers to be run
      #      based on TP degree configured and number of neuron cores/GPUs available.
      #      For example, if TP=2, GPUs=8 then FMBench will start 4 containers and 1 load balancer,
      # auto: only supported if the underlying inference container would automatically 
      #       start multiple copies of the model internally based on TP degree and neuron cores/GPUs
      #       available. In this case only a single container is created, no load balancer is created.
      #       The DJL serving containers supports auto.  
      model_copies: max
      # if you set the model_copies parameter then it is mandatory to set the 
      # tp_degree, shm_size, model_loading_timeout parameters
      tp_degree: 2
      shm_size: 12g
      model_loading_timeout: 2400
    # modify the serving properties to match your model and requirements
    serving.properties: |
      engine=MPI
      option.tensor_parallel_degree=2
      option.max_rolling_batch_size=256
      option.model_id=meta-llama/Meta-Llama-3.1-8B-Instruct
      option.rolling_batch=lmi-dist
```

## Considerations while setting the `model_copies` parameter

1. The `model_copies` parameter is an EC2 only parameter, which means that you cannot use it when deploying models on SageMaker for example.

1. If you are looking for the best (lowest) inference latency then you might get better results with setting the `tp_degree` and `option.tensor_parallel_degree` to the total number of GPUs/Neuron cores available on your EC2 instance and `model_copies` to `max` or `auto` or `1`, in other words, the model is being shared across all accelerators and there can be only 1 copy of the model that can run on that instance (therefore setting `model_copies` to `max` or `auto` or `1` all result in the same thing i.e. a single copy of the model running on that EC2 instance).

1. If you are looking for the best (highest) transaction throughput while keeping the inference latency within a given latency budget then you might want to configure `tp_degree` and `option.tensor_parallel_degree` to the least number of GPUs/Neuron cores on which the model can run (for example for `Llama3.1-8b` that would be 2 GPUs or 4 Neuron cores) and set the `model_copies` to `max`. Let us understand this with an example, say you want to run `Llama3.1-8b` on a `p4de.24xlarge` instance type, you set `tp_degree` and `option.tensor_parallel_degree` to 2 and `model_copies` to `max`, `FMBench` will start 4 containers (as the `p4de.24xlarge` has 8 GPUs) and an Nginx load balancer that will round-robin the incoming requests to these 4 containers. In case of the DJL serving LMI you can achieve similar results by setting the `model_copies` to `auto` in which case `FMBench` will start a single container (and no load balancer since there is only one container) and then the DJL serving container will internally start 4 copies of the model within the same container and route the requests to these 4 copies internally. Theoretically you should expect the same performance but in our testing we have seen better performance with `model_copies` set to `max` and having an external (Nginx) container doing the load balancing.
