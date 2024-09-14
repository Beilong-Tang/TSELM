dev_path=$1
test_path=$2

# Extract files
storage_dir=$(pwd)
tar -xzvf ${dev_path} -C $storage_dir
tar -xzvf ${test_path} -C $storage_dir