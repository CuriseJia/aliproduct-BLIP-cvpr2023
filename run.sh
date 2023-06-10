# conda activate air
cd BLIP/
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
nohup python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py > ../airproduct/train1.log 2>&1 &