has_param() {
    local term="$1"
    shift
    for arg; do
        if [[ $arg == "$term" ]]; then
            return 0
        fi
    done
    return 1
}

# Downloads the dataset
if has_param '--with-data' "$@"; then
    echo "Downloading data"
    ARCHIVE="py150.tar.gz"
    DEST="data"

    curl -O -L -C - http://files.srl.inf.ethz.ch/data/${ARCHIVE}

    echo "Extracting data"
    mkdir -p $DEST
    tar -xf $ARCHIVE -C ./${DEST}

    echo "Cleanup"
    rm -r $ARCHIVE ./${DEST}/parse_python.py

    echo "Data successfully downloaded"
fi

# Creates and activates the conda environment
conda env create -f environment.yml
conda activate code_completion
