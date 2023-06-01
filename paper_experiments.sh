#!/bin/sh

# nohup bash paper_experiments.sh | tee run.log &>/dev/null &

#================ Shared args ================#
times=10
seed=9842365
device="cuda:0"
data_dir="./configs/data/"
model_dir="./configs/model/"
postfix=".yaml"

echo "================================================"
echo "================run Experiment 1================"
echo "================================================"

# ---- model-homogeneous FL
exp_conf="./configs/exp/exp1-hom.yaml"
data=(celeba cifar10 fashion-mnist femnist mnist sent140 synthetic)
hom_algorithms=(Center Local FedAvg FedProx SCAFFOLD FedDF FedRoD FedFTG FedGen FedTED)

for ((t=0; t<times; t+=1)); do
  for alg in ${hom_algorithms[@]}; do
    for d in ${data[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}hom-${d}${postfix} \
      --seed $s --device $device
    done
  done
done

# ---- model-heterogeneous FL
het_algorithms=(Local FedDistill FedMD KT_pFL FedTED)
exp_conf="./configs/exp/exp1-het.yaml"
for ((t=0; t<times; t+=1)); do
  for alg in ${het_algorithms[@]}; do
    for d in ${data[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}het-${d}${postfix} \
      --seed $s --device $device
    done
  done
done

echo "================================================"
echo "================run Experiment 2================"
echo "================================================"
exp_conf="./configs/exp/exp2.yaml"
d=mnist
for ((t=0; t<times; t+=1)); do
  for alg in ${het_algorithms[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}exp2-${d}${postfix} \
      --seed $s --device $device
  done
done

echo "================================================"
echo "================run Experiment 3================"
echo "================================================"
exp_conf="./configs/exp/exp3.yaml"
d=synthetic
alphas=(0.05 0.1 0.5 1.0 5.0)
sigmas=(0.05 0.5 1.0 5.0 10.0)
exp3_algorithms=(Local FedAvg FedRoD FedTED)

for ((t=0; t<times; t+=1)); do
  for alg in ${exp3_algorithms[@]}; do
    for alpha in ${alphas[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}-alpha${alpha}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}hom-${d}${postfix} \
      --seed $s --device $device --exp_name exp3-alpha-${alpha}
    done
    for sigma in ${sigmas[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}-sigma${sigma}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}hom-${d}${postfix} \
      --seed $s --device $device --exp_name exp3-sigma-${sigma}
    done
  done
done

echo "================================================"
echo "================run Experiment 4================"
echo "================================================"
exp_conf="./configs/exp/exp4.yaml"
d=mnist
alg=FedTED

for ((t=0; t<times; t+=1)); do
    for ((n=5; n<65; t+=5)); do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}-100${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}hom-${d}-100${postfix} \
      --seed $s --device $device --num_clients n
    done
done


echo "================================================"
echo "================run Experiment 5================"
echo "================================================"
exp_conf="./configs/exp/exp5.yaml"
d=mnist
algorithms=(FedTED FedTED-FD  FedTED-FD+TN FedTED-N-LP FedTED-RB FedTED-TN)

for ((t=0; t<times; t+=1)); do
  for alg in ${algorithms[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  --exp_conf $exp_conf --data_conf ${data_dir}${d}${postfix} \
      --public_conf ${data_dir}${d}-public${postfix} --model_conf ${model_dir}hom-${d}${postfix} \
      --seed $s --device $device
  done
done
