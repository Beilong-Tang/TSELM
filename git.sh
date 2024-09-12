####################################
## a faster to commit message to git
####################################
# List untracked files and calculate their sizes
# untracked_files=$(git ls-files --others --exclude-standard)

message="$*"

if [ -z "$message" ]; then
  message="update"
else
  :
fi

### check if there are files that are over 100 MB included in the repo
found_large_file=false
git ls-files --others --exclude-standard | while read -r file; do
    if [ -f "$file" ]; then
        file_size=$(du -sh "$file" | awk '{print $1}')
        if [[ $file_size == *M && $(echo "${file_size//M/}" | bc) -gt 50 ]]; then
            echo "Untracked file over 100 MB: $file_size - $file"
            exit 1
        fi
    fi
done || exit 1
# # exit 1
# echo 111222
git add .
git commit -m "$message"
git push