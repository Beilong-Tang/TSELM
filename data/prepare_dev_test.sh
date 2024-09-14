dev_path=$1
test_path=$2
storage_path=$3
# Extract files
tar -xzvf ${dev_path} -C $storage_dir
tar -xzvf ${test_path} -C $storage_dir