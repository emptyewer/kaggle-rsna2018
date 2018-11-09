#!/bin/bash
set -e

function show_usage() {
    echo ""
    echo "Usage: bash $(basename $0) --traindata=<path> --testdata=<path> [--gpus=GPU_IDs] [--share=SHARE_PATH]  [--weights=<path>]"
    echo ""
    echo "   --share: Additional paths to mount in docker."
    echo "            NOTE: supports multiple --share flags."
    echo "   --gpus: GPU-IDs (separated by commas)"
    echo "          default: 0,1,2,3"
    echo ""
    echo "   --traindata: path to the directory of train images"
    echo ""
    echo "   --testdata: path to the directory of test images"
    echo ""
    echo "   --weights: path to the directory of model file(s)"
    echo ""
    echo "   --help: Show this help message."
    echo ""
}

source parser.sh

function main () {
    echo ""
    echo "***** Kaggle Challenge: RSNA 2018 *****"
    printf '%-25s %-10s\n'  "Date:" "$(date +%Y-%m-%d" "%H:%M:%S)"
    printf '%-25s %-10s\n'  "Host:" "$(hostname)"
    echo "***************************************"
    echo ""
    eval $(parse_equal_delimited_params "$@")
    # Display help message if --help flag is set as the first command-line argument
    if [[ "$1" == "--help" || "$1" == "-h" || -z "$traindata" || -z "$testdata" ]]; then
        show_usage
        exit -1
    fi

    CONTAINER_NAME=DeepRadiology_$(date +%d%b%Y)-r${RANDOM}
    LOCAL_USER_ID=$(id -u ${USER})
    LOCAL_GROUP_ID=$(id -g ${USER})
    DOCKER_IMAGE=deepradiology/kaggle:rsna2018
    GPU=${gpus-0,1,2,3}

    docker pull $DOCKER_IMAGE

    # get mount string for all shared paths
    SHARE_ARGS=""
    for s in "${share[@]}"; do
        SHARE_ARGS="$SHARE_ARGS -v $s:$s "
    done

    # data path
    SHARE_ARGS="$SHARE_ARGS -v $traindata:/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTrain "
    SHARE_ARGS="$SHARE_ARGS -v $testdata:/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTest "

    if [ -z $weights ]; then
        weights=$(readlink -m weights)
        if [ ! -d $weights ]; then
            mkdir $weights
        fi
    fi

    if [ ! -d submissions ]; then
        mkdir submissions
    fi

    if [ ! -d ensemble ]; then
        mkdir ensemble
    fi

    # model path
    SHARE_ARGS="$SHARE_ARGS -v $weights:/notebooks/save/couplenet/res152/kaggle_pna "

    SHARE_ARGS="$SHARE_ARGS -v $(pwd)/submissions:/notebooks/output/couplenet/res152/kaggle_pna "

    SHARE_ARGS="$SHARE_ARGS -v $(pwd)/ensemble:/notebooks/ensemble "
    SHARE_ARGS="$SHARE_ARGS --shm-size=64G "
    export NV_GPU=$GPU
    RUN_OPTS+="--rm -d -p 2222:8888 --name ${CONTAINER_NAME} "
    printf '%-25s %-10s%s\n'  "Volume maps:" "$SHARE_ARGS"
    printf '%-25s %-10s%s\n'  "Run Opts:" "$RUN_OPTS"
    printf "\n\n"
    RUN_OPTS+="$SHARE_ARGS "

    teardown() {
      trap - SIGINT SIGTERM SIGKILL
      echo "$(date +%Y-%m-%d" "%H:%M:%S) Stopping container ${CONTAINER_NAME}"
      docker stop -t1 ${CONTAINER_NAME}
      echo "$(date +%Y-%m-%d" "%H:%M:%S) Container ${CONTAINER_NAME} stopped."
      echo ""
    }
    trap teardown SIGINT SIGTERM SIGKILL

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting DeepRadiology Notebook ..."
    CONTAINER_ID=$(nvidia-docker run ${RUN_OPTS} ${DOCKER_IMAGE} /bin/bash -c "chown -R ${LOCAL_USER_ID}:${LOCAL_GROUP_ID} /work; chown -R ${LOCAL_USER_ID}:${LOCAL_GROUP_ID} /home/user; chown -R ${LOCAL_USER_ID}:${LOCAL_GROUP_ID} /notebooks; chown -R ${LOCAL_USER_ID}:${LOCAL_GROUP_ID} /opt/R-FCN.pytorch; tail -f /dev/null")
    echo "Container ID: ${CONTAINER_ID} (${CONTAINER_NAME})"
    nvidia-docker exec -u ${LOCAL_USER_ID}:${LOCAL_GROUP_ID} -e HOME=/home/user ${CONTAINER_NAME} /run_jupyter.sh
}

if (( $# < 1 )); then
    show_usage
    exit -1
fi

main "$@"
exit 0
