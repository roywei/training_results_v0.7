RANK=$1
NODES=$2
OUTPUT=$3
MASTER=$4
BASE_LR=0.16
MAX_ITER=40000
WARMUP_FACTOR=0.000256
WARMUP_ITERS=625
STEPS="\"(9000,12000)\""
TRAIN_IMS_PER_BATCH=128
TEST_IMS_PER_BATCH=128
FPN_POST_NMS_TOP_N_TRAIN=4000
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=24
NPROC_PER_NODE=8
if [ $RANK = 0 ]
then
	docker run -it --rm --gpus all --name mlperf_training \
        --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
        --ulimit=stack=67108864 --ulimit=memlock=-1 \
        -v /workspace/data:/workspace/object_detection/datasets mrcnn /bin/bash -c \
        "wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl -P /root/.torch/models/ && \
         python -u -m bind_launch --nnodes $NODES --node_rank $RANK --master_addr $MASTER --master_port 1234 --nsockets_per_node=${NSOCKETS_PER_NODE} \
                      		  --ncores_per_socket=${NCORES_PER_SOCKET} --nproc_per_node=${NPROC_PER_NODE} \
				  tools/train_mlperf.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x_pisa.yaml' \
				  	DTYPE 'float16' \
					PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog.py' \
					DISABLE_REDUCED_LOGGING True \
					SOLVER.BASE_LR ${BASE_LR} \
                                        SOLVER.MAX_ITER ${MAX_ITER} \
                                        SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
                                        SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
                                        SOLVER.WARMUP_METHOD mlperf_linear \
					SOLVER.STEPS $STEPS \
                                        SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
                                        TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
                                        MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
                                        MODEL.ROI_BOX_HEAD.CARL False \
                                        MODEL.ROI_BOX_HEAD.ISR_P False \
                                        MODEL.ROI_BOX_HEAD.ISR_N False \
                                        MODEL.ROI_BOX_HEAD.DECODE False \
                                        MODEL.ROI_BOX_HEAD.LOSS "SmoothL1Loss" \
                                        MODEL.ROI_BOX_HEAD.PISA_ONEPASS False \
                                        NHWC True" | tee $OUTPUT
else
	docker run -it --rm --gpus all --name mlperf_training \
        --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
        --ulimit=stack=67108864 --ulimit=memlock=-1 \
        -v /workspace/data:/workspace/object_detection/datasets mrcnn /bin/bash -c \
        "wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl -P /root/.torch/models/ && \
	 python -u -m bind_launch --nnodes $NODES --node_rank $RANK --master_addr $MASTER --master_port 1234 --nsockets_per_node=${NSOCKETS_PER_NODE} \
                                  --ncores_per_socket=${NCORES_PER_SOCKET} --nproc_per_node=${NPROC_PER_NODE} \
                                  tools/train_mlperf.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x_pisa.yaml' \
                                        DTYPE 'float16' \
                                        PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog.py' \
                                        DISABLE_REDUCED_LOGGING True \
                                        SOLVER.BASE_LR ${BASE_LR} \
                                        SOLVER.MAX_ITER ${MAX_ITER} \
                                        SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
                                        SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
                                        SOLVER.WARMUP_METHOD mlperf_linear \
                                        SOLVER.STEPS $STEPS \
                                        SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
                                        TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
                                        MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
                                        MODEL.ROI_BOX_HEAD.CARL False \
                                        MODEL.ROI_BOX_HEAD.ISR_P False \
                                        MODEL.ROI_BOX_HEAD.ISR_N False \
                                        MODEL.ROI_BOX_HEAD.DECODE False \
                                        MODEL.ROI_BOX_HEAD.LOSS "SmoothL1Loss" \
                                        MODEL.ROI_BOX_HEAD.PISA_ONEPASS False \
                                        NHWC True"
fi
