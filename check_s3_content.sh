# this scripts creates a local directory for running FMBench
# without any s3 dependency and copies all relevant files
# for public FMBench content
#  RUN THIS SCRIPT AS A SANITY CHECK BEFORE EVERY NEW RELEASE

# sanity check to confirm that all config files and other files listed in manifest.txt are 
# indeed present in the blogs bucket. If a file is not present in the blogs bucket then the
# CloudFormation template would error out.
BUCKET=aws-blogs-artifacts-public
CONFIGS_ONLY=configs
for i in `cat manifest.txt | grep $CONFIGS_ONLY`
do
  wget -q --spider https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/$i --no-check-certificate
  result=$?
  if [ $result -eq 0 ]; then
    #echo $i exists in the ${BUCKET}
    : # noop
    # check if this file is differnet from what we have in the repo
    # if so then we need to flag for uploading the latest version to
    # the bucket
    # compare the contents of the downloaded file with the local file
    TEMP_FILE=/tmp/tempfile
    REMOTE_FILE=https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/$i
    wget -q $REMOTE_FILE --no-check-certificate -O $TEMP_FILE
    
    LOCAL_FILE=src/fmbench/$i
    if diff "$TEMP_FILE" "$LOCAL_FILE" -w> /dev/null; then
        #echo "The contents of the remote file and the local file are identical."
        : # noop
    else
        echo "contents of the $REMOTE_FILE and the local $LOCAL_FILE are different, needs to be updated manually"
    fi
    rm -f $TEMP_FILE

  else
    echo $i does not exist in the ${BUCKET}, copy it there manually
  fi
done
