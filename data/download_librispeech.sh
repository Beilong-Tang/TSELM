#1. Download LibriSpeech train-clean[100,360]
storage_dir=$1

mkdir -p $storage_dir
function LibriSpeech_clean100() {
	if ! test -e $librispeech_dir/train-clean-100; then
		echo "Download LibriSpeech/train-clean-100 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-100.tar.gz
	fi
}

function LibriSpeech_clean360() {
	if ! test -e $librispeech_dir/train-clean-360; then
		echo "Download LibriSpeech/train-clean-360 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-360.tar.gz
	fi
}

# Download dataset
LibriSpeech_clean100 & LibriSpeech_clean360


#2. Download Libri2Mix auxilary dataset





