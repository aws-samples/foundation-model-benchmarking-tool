# this scripts creates a local directory for running FMBench
# without any s3 dependency and copies all relevant files
# for public FMBench content
FMBENCH_READ_DIR=/tmp/fmbench-read
BUCKET=aws-blogs-artifacts-public

mkdir $FMBENCH_READ_DIR
wget https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/manifest.txt -P ${FMBENCH_READ_DIR}/

# copy each file of the public content for FMBench
for i in `cat ${FMBENCH_READ_DIR}/manifest.txt`
do
  dir_path=`dirname $i`
  mkdir -p ${FMBENCH_READ_DIR}/$dir_path
  wget https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/$i -P ${FMBENCH_READ_DIR}/$dir_path
done


