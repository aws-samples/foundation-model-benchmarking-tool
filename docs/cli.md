# `FMBench` CLI

Here are the command line options currently supported by the `fmbench` CLI.

```{.bashrc}
usage: fmbench [-h] --config-file CONFIG_FILE [--role-arn ROLE_ARN] [--local-mode {yes,no}] [--tmp-dir TMP_DIR] [--write-bucket WRITE_BUCKET] -A [key=value]

Run FMBench with a specified config file.
```

options:  
  `-h`, `--help`            show this help message and exit  

  `--config-file` CONFIG_FILE
                        The S3 URI of your Config File  

  `--role-arn` ROLE_ARN   (_Optional_) The ARN of the role to be used for FMBench. If an Amazon SageMaker endpoint is being deployed through FMBench then this role would also be used by that endpoint  

  `--local-mode` {yes,no}  Specify if running in local mode or not. Options: yes, no. Default is no.  

  `--tmp-dir` TMP_DIR    (_Optional_)  An optional tmp directory if fmbench is running in local mode.  

  `--write-bucket` WRITE_BUCKET  Write bucket that is used for sagemaker endpoints in local mode and storing metrics in s3 mode.  

  `-A` key=value        (_Optional_) Specify a key value pair which will be used to replace the `{key}` in the given config file. The key could be anything that you have templatized in the config file as `{key}` and it will be replaced with `value`. This comes in handy when using a generic configuration file and replace keys such `model_id`, `tp_degree` etc. Note that you are not limited to pre-defined set of keys, you can put any key in the config file as `{key}` and it will get replaced with its value. If there are multiple key value pairs, then simply specify the `-A` option multiple times in the command line.  


