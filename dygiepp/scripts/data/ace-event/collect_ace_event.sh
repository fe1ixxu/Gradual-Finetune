# Collect all the ace data into one directory, for convenience

ace_dir=$1
lang=$2

original_dir=$ace_dir/data/${lang}

out_dir=./data/ace-event/raw-data


mkdir -p $out_dir

ls $original_dir |
    while read subdir
    do
        this_dir=$original_dir/$subdir/adj
        cp $this_dir/*apf.xml $out_dir
        cp $this_dir/*.sgm $out_dir
    done