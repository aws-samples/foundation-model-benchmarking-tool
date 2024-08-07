#!/bin/sh
set -e
# download model from HuggingFace -> compile it for Neuron -> deploy on SageMaker
# param 1: HuggingFace token
# param 2: HuggingFace model id, for example meta-llama/Meta-Llama-3-8B-Instruct
# param 3: local directory path to save the model

## split and save the model 
token=$1
model_id=$2
neuron_version=$3
model_store=$4
s3_bucket=$5
prefix=$6
region=$7
role=$8
batch_size=$9
num_neuron_cores=${10}
ml_instance_type=${11}
model_loading_timeout=${12}
serving_properties=${13}
script_path=${14}
instance_count=${15}
ml_image_uri=${16}
model_id_wo_repo=`basename $2`
model_id_wo_repo_split=$model_id_wo_repo-split
local_dir=neuron_version/$neuron_version/$model_store/$model_id_wo_repo/$model_id_wo_repo_split
export HF_TOKEN=$token

echo model_id_wo_repo=$model_id_wo_repo, model_id_wo_repo_split=$model_id_wo_repo_split
echo model_id=$model_id, local_dir=$local_dir, neuron_version=$neuron_version, model_store=$model_store
echo s3_bucket=$s3_bucket, prefix=$prefix, region=$region, role=$role
echo batch_size=$batch_size, num_neuron_cores=$num_neuron_cores, ml_instance_type=$ml_instance_type
echo HF_TOKEN=$token
echo script_path=$script_path

# download the model
echo going to download model_id=$model_id, local_dir=$local_dir
echo Going into split and save with HF_token=$token
python $script_path/scripts/split_and_save.py --model-name $model_id --save-path $local_dir 
echo model download step completed

#LLama3 tokenizer fix
tokenizer_config_json=`find . -name tokenizer_config.json`
sed -i 's/end_of_text/eot_id/g' $tokenizer_config_json

#"../2.18/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
# compile the model
echo starting model compilation...
python $script_path/scripts/compile.py --action compile --batch-size $batch_size --num-neuron-cores $num_neuron_cores --model-dir $local_dir 
echo done with model compilation

# now upload the model binaries to the s3 bucket
echo going to upload from neuron_version/$neuron_version/$4/ to s3://$s3_bucket/$prefix/
aws s3 cp --recursive neuron_version/$neuron_version/$model_store/ s3://$s3_bucket/$prefix/
echo done with s3 upload

# dir for storing model artifacts
model_dir=smep-with-lmi/models/$model_id
mkdir -p $model_dir
# prepare serving.properties
serving_prop_fpath=$model_dir/serving-inf2.properties
cat << EOF > $serving_prop_fpath
$serving_properties
EOF

# prepare model packaging script
model_packaging_script_fpath=$model_dir/package-inf2.sh
cat << EOF > $model_packaging_script_fpath
mkdir mymodel
cp serving-inf2.properties  mymodel/serving.properties
tar czvf mymodel-inf2.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-inf2.tar.gz s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/${model_id_wo_repo_split}/code/
EOF
chmod +x $model_packaging_script_fpath

# now change director to the model dir we just created and run
# the above model packaging script which creates a model.tar.gz that has
# the serving.properties which in turn contains the model path in s3 and 
# other model parameters.
cd $model_dir
echo now in `pwd`
./package-inf2.sh
cd -
echo now back in `pwd`

echo near the end of the script, we will deploy the model
python $script_path/smep-with-lmi/deploy.py --device inf2 \
  --aws-region $region \
  --role-arn $role \
  --bucket $s3_bucket \
  --model-id $model_id \
  --prefix $prefix \
  --inf2-instance-type $ml_instance_type \
  --inf2-image-uri $ml_image_uri \
  --model-s3-uri s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/${model_id_wo_repo_split}/code/mymodel-inf2.tar.gz \
  --neuronx-artifacts-s3-uri s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/neuronx_artifacts \
  --script-path $script_path \
  --model-data-timeout $model_loading_timeout \
  --initial-instance-count $instance_count

echo all done

