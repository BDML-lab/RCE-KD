"""
For Recommendation Models
"""
# train teacher
python -u main.py --dataset=citeulike --S_backbone=bpr --train_teacher --suffix teacher
python -u main.py --dataset=citeulike --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=citeulike --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --dataset=gowalla --S_backbone=bpr --train_teacher --suffix teacher
python -u main.py --dataset=gowalla --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=gowalla --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --dataset=yelp --S_backbone=bpr --train_teacher --suffix teacher --suffix teacher
python -u main.py --dataset=yelp --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=yelp --S_backbone=hstu --train_teacher --suffix teacher --postsave

# from scratch
python -u main.py --dataset=citeulike --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=citeulike --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=citeulike --S_backbone=hstu --model=scratch --suffix student

python -u main.py --dataset=gowalla --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=gowalla --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=gowalla --S_backbone=hstu --model=scratch --suffix student

python -u main.py --dataset=yelp --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=yelp --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=yelp --S_backbone=hstu --model=scratch --suffix student

# KD
# For HetComp, you need pre-save teacher checkpoints through:
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --train_teacher --no_log --ckpt_interval=50
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=hetcomp
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rrd
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=dcd
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rcekd

python -u main.py --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=de
python -u main.py --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=rrd


python -u main.py --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=rrd


python -u main.py --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=rrd
