#!/bin/zsh

# change with -d
dataloc=./data

# change with -u
cpucount=$(lscpu | grep ^CPU\(s\): | sed -e 's|.* ||')

function run_print() {
  echo "[[RUNSCRIPT]] $1"
}

function quit_print() {
  echo "[[TERMINATING RUNSCRIPT]] $1, EXITING"
  exit
}

function get_gpu() {
  dev=0

  # # UNCOMMENT SECTION BELOW TO SELECT A FREE GPU WHEN MULTIPLE ARE AVAILABLE
  # # yes this is really ugly :(
  # nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp
  # dev=$(echo -e 'import numpy as np\nprint(np.argmax([int(x.split()[2]) for x in open("tmp", "r").readlines()]))' | python)
  # rm tmp

  echo $dev
}

# default configuration: imagenet, greybox
# configuration used to generate the results presented in the paper
dataset="imagenet"  # use cifar with -s
# scales=(2.0 2.5 3.0 3.5 4.0 5.0)  # imagenet scales configuration
scales=(1.0 2.0 2.5 3.0 4.0)  # imagenet scales configuration
biases=(1.0)  # sweep with -b
cutoff=20
range=18
batchsize=64  # change if seeing out of memory errors on GPU
latentdim=110
target=757
trainbatch=10

# remain the same for both datasets
weightdecay="1e-4"
lr=0.0001
epochs=50

bbox=""
base=""
vbase=""
# set queries to test queries instead of testing scales
QUERIES=false

ckpt=""
queries=(full)  # no limit

GANONLY=false

stepsize=200

# TODO add help message
while getopts s:bc:xaeqt:d:u:gf:h: arg;
do
  case $arg in
    s)
      dataset=$OPTARG
      scales=(0.5 0.75)
      batchsize=256
      latentdim=100
      target=0
      stepsize=200
      if [ $dataset = "cifar10_pgd" ];
      then
        stepsize=200
        cutoff=3
        range=2
      else
        cutoff=20
        range=18
      fi
      trainbatch=10
      ;;

    b)  # data limits sweep; max_triggers
      scales=(2.5)
      biases=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 1.0)
      ;;

    c)  # specify checkpoint instead of generating
      if [ -f "$OPTARG" ]
      then
        ckpt=$OPTARG
      else
        quit_print "$OPTARG does not exist"
      fi
      ;;

    x)  # use black box
      bbox="--bbox"
      ;;

    a)  # no-info GAN baseline
      base="--base"
      ;;

    e)  # noise only baseline
      vbase="--baseline"
      base="--base"
      scales=(0.1 0.25 0.5 1.0 2.0)
      ;;

    q)  # sweep queries instead of scales
      QUERIES=true
      queries=(20 40 100 200 full)
      ;;

    t)  # target label
      target=$OPTARG
      ;;

    d)  # location where datasets are saved
      dataloc=$OPTARG
      ;;

    u)  # number of cores to use during training
      cpucount=$OPTARG
      ;;

    g)  # train trigger generator, then quit
      GANONLY=true
      ;;

    f)
      cutoff=$OPTARG
      ;;

    h)
      range=$OPTARG
      ;;

    \?)
      exit
      ;;
  esac
done

if [ ! -d $dataloc ]
then
  quit_print "$dataloc not found. Please create it or specify it with -d"
fi

# display run info
run_print "Using $dataset from $dataloc with $cpucount CPUs"
run_print "Target is $target"
if [ -n "$bbox" ];
then
  run_print "Using a blackbox GAN"
elif [ -n "$base" ];
then
  run_print "Training a baseline"
#  queries=(10 20 30 40 100 200 full)
fi

if [ -n "$vbase" ]
then
  run_print "Checkpoint is a placeholder for compatibility; it's not used in this mode"
  queries=(full)
fi

if [ $QUERIES = true ];
then
  if [ -n "$bbox" ];
  then
    run_print "It's suggested that queries is used with blackbox"
  fi

  # limit scales under queries sweep
  if [ $dataset = "imagenet" ];
  then
    scales=(2.5)
  else
    run_print 'sweeping all'
#    scales=(0.75)
  fi
fi

# if [ -n "$bbox" ]
# then
#   if [ $dataset = "imagenet" ];
#   then
#     cutoff=$cutoff
#   else  # bbox cifar
#     cutoff=15
#     range=5
#   fi
# fi


# Generate checkpoint, if none specified
if [ -z "$ckpt" ];
then
  full_ckpt="$dataset"_t"$target"_full_generator.ckpt

  if [ -n "$bbox" ]
  then
    full_ckpt=bbox_"$full_ckpt"
  elif [ -n "$base" ]
  then
    full_ckpt=base_"$full_ckpt"
  fi

  full_ckpt="$dataset"_sweep_res/"$full_ckpt"

  if [ -f "$full_ckpt" ];
  then
    run_print "Found $full_ckpt"
  else
    run_print "Training Trigger Generator $full_ckpt"

    if [ -n "$bbox" ]
    then
      flags="--bbox_loss"
    elif [ -n "$base" ]
    then
      flags="--base_loss"
    else
      flags="--margin_loss"
    fi

    name=$full_ckpt

    gpu=$(get_gpu)
    run_print "Using GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python \
    acgan.py --dataset $dataset --batch-size $trainbatch --lr 0.002 \
      --sample-interval 100 --step-size "$stepsize" --n-epochs 400 \
      --latent-dim $latentdim --target "$target" --cutoff "$cutoff" \
      --cutoff_range "$range" --num_workers $cpucount --data-dir "$dataloc" \
      --idx-dir "$dataloc" "$flags" --output_directory "$dataset"_sweep_res \
      || quit_print 'python error'
  fi
else
  if [ $QUERIES = true ];
  then
    quit_print "Specifying checkpoint does not work with query limits"
  fi
fi

# set checkpoint name
if [ -z "$ckpt" ];
then
  ckpt="$dataset"_t"$target"_full_generator.ckpt
  if [ -n "$bbox" ];
  then
    ckpt=bbox_"$ckpt"
  elif [ -n "$base" ]
  then
    ckpt=base_"$ckpt"
  fi
fi

if [ -f "$ckpt" ]
then
  run_print "Using $ckpt"
else
  ckpt="$dataset"_sweep_res/"$ckpt"
  run_print "Set ckpt path as $ckpt"
fi

run_print "Evaluating thresholds for $ckpt"

if [ -n "$vbase" ]
then
  gpu=$(get_gpu)
  run_print "Using GPU $gpu"

  CUDA_VISIBLE_DEVICES=$gpu python \
  threshold_perf.py --dataset "$dataset" --clip 0.1 \
    --batch_size 50 --num_workers $cpucount \
    --output_directory "$dataset"_sweep_res/"$cutoff"-"$range"-thresh \
    --train_set --data-dir "$dataloc" --idx-dir "$dataloc" $ckpt --baseline \
    || quit_print 'python error'
else
  gpu=$(get_gpu)
  run_print "Using GPU $gpu"

  CUDA_VISIBLE_DEVICES=$gpu python \
  threshold_perf.py --dataset "$dataset" --clip 0.1 \
    --batch_size 50 --num_workers $cpucount \
    --output_directory "$dataset"_sweep_res/"$cutoff"-"$range"-thresh \
    --train_set --data-dir "$dataloc" --idx-dir "$dataloc" $ckpt \
    || quit_print 'python error'
fi

if [ $GANONLY = true ];
then
  exit
fi


run_print "Sweeping [${biases[*]}] x [${scales[*]}] x [${queries[*]}]"

for bias in ${biases[@]}; do
  for scale in ${scales[@]}; do
    for limit in ${queries[@]}; do
      name="$dataset-$bias-$scale-$limit"
      if [ -n "$bbox" ];
      then
        name="bbox-$name"
      elif [ -n "$vbase" ]
      then
        name="noise-$name"
      elif [ -n "$base" ]
      then
        name="base-$name"
      fi

      run_print "Running $name"

      run_print "Using $ckpt"

      # train on the generated trojan
      gpu=$(get_gpu)
      run_print "Using GPU $gpu"

      CUDA_VISIBLE_DEVICES=$gpu python \
      train_on_gen.py $bias $scale "$ckpt" --run-info "$name" \
        --dataset $dataset --batch_size $batchsize \
        --weight-decay $weightdecay --num_workers $cpucount \
        --lr $lr --epochs $epochs --data-dir "$dataloc" $vbase \
        --output_directory "$dataset"_sweep_res \
        --idx-dir "$dataloc" --max-scale 3 || quit_print 'python error'

    done
  done
done
