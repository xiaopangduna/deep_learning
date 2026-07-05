#!/bin/bash



export VSI_NN_ENABLE_OPCHECK=0


if [ -z "$ACUITY_PATH" ]; then
    echo "Need to set environment variable ACUITY_PATH"
    exit 1
fi

PEGASUS=$ACUITY_PATH/pegasus
if [ ! -e "$PEGASUS" ]; then
    PEGASUS="python3 $PEGASUS.py"
fi


function convert_platform_to_optimize()
{
    PLATFORM=$1
    echo "convert PLATFORM=${PLATFORM}"

    if [ ${PLATFORM} = 'v85x' ] || [ ${PLATFORM} = 'v853' ]; then
        OPTIMIZE=VIP9000PICO_PID0XEE
    elif [ ${PLATFORM} = 'r853' ]; then
        OPTIMIZE=VIP9000PICO_PID0XEE
    elif [ ${PLATFORM} = 'mr527' ]; then
        OPTIMIZE=VIP9000NANOSI_PLUS_PID0X10000016
    elif [ ${PLATFORM} = 't527' ]; then
        OPTIMIZE=VIP9000NANOSI_PLUS_PID0X10000016
    elif [ ${PLATFORM} = 'mr536' ] || [ ${PLATFORM} = 't536' ]; then
        OPTIMIZE=VIP9000NANODI_PLUS_PID0X1000003B
    elif [ ${PLATFORM} = 'a733' ]; then
        OPTIMIZE=VIP9000NANODI_PLUS_PID0X1000003B
    elif [ ${PLATFORM} = 't736' ]; then
        OPTIMIZE=VIP9000NANODI_PLUS_PID0X1000003B
    else
        echo "=========== wrong platform ! ( v853  / r853  / mr527 / t527)==========="
        echo "=========== wrong platform ! ( mr536 / t536  / a733  / t736)==========="
        exit -1
    fi
}

function export_ovx_network()
{
    NAME=$1
    pushd $NAME

    QUANTIZED=$2
    PLATFORM_NAME=$3
    OPTIMIZED=$4

    if [ ${QUANTIZED} = 'float' ]; then
        TYPE=float
        generate_path='./wksp/${NAME}_fp16'
    else
        TYPE=quantized
        generate_path='./wksp/${NAME}_${QUANTIZED}_hybrid'
    fi

    echo " ======================================================================="
    echo " =========== Start Generate $NAME ovx C code with type of ${QUANTIZED} ==========="
    echo " ======================================================================="


    # if want to import c code into win IDE , change --target-ide-project command-line param from 'linux64' -> 'win32'
    if [ ${QUANTIZED} = 'float' ]; then
        cmd="$PEGASUS export ovxlib \
            --pack-nbg-unify                                \
            --optimize              ${OPTIMIZED}            \
            --viv-sdk               ${VIV_SDK}   \
            --model                 ${NAME}.json \
            --model-data            ${NAME}.data \
            --dtype                 ${TYPE} \
            --target-ide-project    'linux64'\
            --with-input-meta       ${NAME}_inputmeta.yml \
            --postprocess-file      ${NAME}_postprocess_file.yml \
            --output-path           ${generate_path}/${NAME}_fp16"
    else

        if [ -f ${NAME}_${QUANTIZED}.quantize ]; then
            echo -e "\033[31m using  ${NAME}_${QUANTIZED}.quantize \033[0m"
        else
            echo -e "\033[31m Can not find  ${NAME}_${QUANTIZED}.quantize \033[0m"
            exit -1;
        fi

        cmd="$PEGASUS export ovxlib \
            --pack-nbg-unify                                \
            --optimize              ${OPTIMIZED}            \
            --viv-sdk               ${VIV_SDK}   \
            --model                 ${NAME}_${QUANTIZED}_hybrid.quantize.json \
            --model-data            ${NAME}.data \
            --dtype                 ${TYPE} \
            --model-quantize        ${NAME}_${QUANTIZED}_hybrid.quantize\
            --target-ide-project    'linux64'\
            --with-input-meta       ${NAME}_inputmeta.yml \
            --postprocess-file      ${NAME}_postprocess_file.yml \
            --output-path           ${generate_path}/${NAME}_${QUANTIZED}"
    fi

    echo $cmd
    eval $cmd

    echo " ======================================================================="
    echo " =========== End  Generate $NAME ovx C code with type of ${quantization_type} ==========="
    echo " ======================================================================="


    mv_cmd="mv ${generate_path}_nbg_unify/network_binary.nb ${generate_path}_nbg_unify/${NAME}_${QUANTIZED}_hybrid_${PLATFORM_NAME}.nb"
    eval $mv_cmd
    echo " =========== End rename network_binary.nb to ${NAME}_${QUANTIZED}_hybrid_${PLATFORM_NAME}.nb ==========="

    cp_cmd="cp ${generate_path}_nbg_unify/${NAME}_${QUANTIZED}_hybrid_${PLATFORM_NAME}.nb ../model/"
    eval $cp_cmd
    echo " =========== copy ${NAME}_${QUANTIZED}_hybrid_${PLATFORM_NAME}.nb to ../model/ ==========="


    rm_cmd="rm -rf ./wksp/"
    eval $rm_cmd

    popd
}

if [ "$#" -lt 3 ]; then
    echo -e "\033[31m  Enter three parameters \033[0m"
    echo "Usage: pegasus_export_ovx_nbg-hybrid.sh <model_name> <quantize_type> <platform>"
    echo "quantize_type    : uint8 / pcq / int16 / float "
    echo "platform         : v853 / r853 / mr527 / t527 / mr536 / t536 / a733 / t736 "
    exit -1
fi


convert_platform_to_optimize ${3%/}

export_ovx_network ${1%/} ${2%/} ${3%/} ${OPTIMIZE}



#./pegasus_export_ovx_post.sh