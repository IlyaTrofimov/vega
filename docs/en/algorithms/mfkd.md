# Multi-fidelity Neural Architecture Search with Knowledge Distillation

## Algorithm Introduction

https://arxiv.org/pdf/2006.08341.pdf

Neural architecture search (NAS) targets at finding the optimal architecture of a neural network for a problem or a family of problems. Evaluations of neural architectures are very time-consuming. One of the possible ways to mitigate this issue is to use low-fidelity evaluations, namely training on a part of a dataset, fewer epochs, with fewer channels, etc. In this paper, we propose to improve low-fidelity
evaluations of neural architectures by using a knowledge distillation. Knowledge distillation adds to a loss function a term forcing a network to mimic some teacher network. The training on the small part of a dataset with such a modified loss function leads to a better selection of neural architectures
than training with a logistic loss. The proposed low-fidelity evaluations were incorporated into a multi-fidelity search algorithm that outperformed the search
based on high-fidelity evaluations only (training on a full dataset).

## Algorithm Principles

The library includes two algorithms:

1) MFKD1 uses low-fidelity evaluations with KD only. Several architectures are sampled randomly from the search space, trained for on a small random subset. Then the GPR regression is fitter to predict the testing accuracy of a network. Finally, the architecture from the whole search space is selected by maximum predicted accuracy.
2) MFKD2 combines low-fidelity and high-fidelity evaluations (training on the subset and full dataset) in the multi-fidelity algorithm. The MFKD2 algorithm
does sequentially two series of steps for low- and high-fidelity evaluations, using the optimum of the former one as an initial point for the later one 

## Search Space 
User can specify any search space, in examples we used the MobileNetV2 search space with various numbers of block repetitions and channels count per layer.

```yaml
search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            name: MobileNetV2
            num_classes: 100
            layer_0:
                repetitions: [1, 2, 3, 4]
                channels: [16, 24]
            layer_1:
                repetitions: [1, 2, 3, 4]
                channels: [24, 32]
            layer_2:
                repetitions: [1, 2, 3, 4]
                channels: [32, 64]
            layer_3:
                repetitions: [1, 2, 3, 4]
                channels: [64, 96]
            layer_4:
                repetitions: [1, 2, 3, 4]
                channels: [96, 160]
            layer_5:
                repetitions: [1, 2, 3, 4]
                channels: [160, 320]
            layer_6:
                repetitions: [1, 2, 3, 4]
                channels: [320, 640]
```
## Usage Guide

Firsly, one needs to train a teacher network on the dataset of interest, here is an example for MobileNetV2 ```examples/nas/mfkd/train_teacher.yml```.
Then run MFKD1 or MFKD2 (```examples/nas/mfkd/mfkd1.yml```, ```examples/nas/mfkd/mfkd2.yml```).
In both of them a user should specify the path to the teacher model and the number of classes in the dataset:
```yaml
trainer:
        type: MFKDTrainer
        teacher: '/home/trofim/mobilenetv2_cifar100_teacher.pth'
        teacher_num_classes: 100
```

### Example 1: MFKD1 Algorithms

```yaml
    search_algorithm:
        type: MFKD1
        max_samples: 30
        seed: 7
    trainer:
        type: MFKDTrainer
        teacher: '/home/trofim/mobilenetv2_cifar100_teacher.pth'
        teacher_num_classes: 100
        epochs: 100
        valid_freq: 10
        optim:
            type: SGD
            lr: 0.1
            momentum: 0.9
            weight_decay: 5.0e-4
        lr_scheduler:
            type: CosineAnnealingLR
            T_max: 100
        loss:
            type: NSTLoss
            beta: 12.5
```

### Example 2: MFKD2 Algorithm

The MFKD2 algorithm uses two levels of fidelity specified by sequencial pipelines:

```yml
pipeline: [nas_low_fidelity, nas_high_fidelity]
```

Parameters of the first pipeline (low-fidelity):

```yml
    search_algorithm:
        type: MFKD2
        init_samples: 5
        max_samples: 10
    trainer:
        type: MFKDTrainer
        teacher: '/home/trofim/mobilenetv2_cifar100_teacher.pth'
        teacher_num_classes: 10
        epochs: 100
        valid_freq: 10
        optim:
            type: SGD
            lr: 0.1
            momentum: 0.9
            weight_decay: 5.0e-4
        lr_scheduler:
            type: CosineAnnealingLR
            T_max: 100
        loss:
            type: KDLoss
            T: 32
            alpha: 1
```

Parameters of the last pipeline (high-fidelity):

```yml
    search_algorithm:
        type: MFKD2
        init_samples: 5
        max_samples: 10
    trainer:
        type: Trainer
        epochs: 100
        valid_freq: 10
        optim:
            type: SGD
            lr: 0.1
            momentum: 0.9
            weight_decay: 5.0e-4
        lr_scheduler:
            type: CosineAnnealingLR
            T_max: 100
```


### Algorithm output

- The optimal models with fully training.
- Logs of all models during the entire search process, and logs for models from the Pareto front(pareto_front.csv).

## Benchmark

Benchmark configuration: [sp_nas.yml](https://github.com/huawei-noah/vega/tree/master/examples/nas/sp_nas.yml)
