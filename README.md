# TOAD-GAN evaluation


The code here has been adapted from the official pytorch implementation of the paper: "TOAD-GAN: Coherent Style Level Generation from a Single Example"
For more information on TOAD-GAN, please refer to the paper ([arxiv](https://arxiv.org/pdf/2008.01531.pdf)). That implementation is available at https://github.com/Mawiszus/TOAD-GAN.

## Getting Started

In order to run the evaluations provided here, please visit the original TOAD-GAN repository as linked to above and follow the instructions for setup given there. In order to use the virtual display and multiprocessing options provided here you will also need to install ray and xvfbwrapper. Both of these libraries can be installed using pip.

In order to run these evaluations, a TOAD-GAN generator must already be trained and available. This can be done following the instructions available at the original TOAD-GAN repo.

## Performing Evaluations

To run the type evaluations used in our paper, the below command can be used:
```
$ python evaluate.py --input-dir input --input-name lvl_1-1.txt --obj {OBJECTIVE} --bcs {BEHAVIORAL CHARACTERISTICS 1} {BEHAVIORAL CHARACTERISTICS 2} --not_cuda --vdisplay --multiproc
```
This will save an archive and heatmaps in a numbered folder under the logs_cmaes subfolder. 

Available Objectives are as follows: "pipes_and_platform", enemy_on_stairs", "new_tiles", "pipes", "koopas", "enemies", "ncd", "hamming", "playable", "tpkl", "unplayable", "platform".

Available behavioral characteristics are as follows: "spiky", "enemy_on_stairs", "new_tiles", "pipes", "koopas", "enemies", "ncd", "hamming", "playable", "tpkl", "time", "max_jump", "n_jumps", "n_enemies", "platform".
    

The following command can be used to generate level images for all vectors saved in an archive in all experiments
```
$ python generate_all.py --not_cuda
```
The following command can be used to generate level images for all vectors saved in an archive from a specific experiment:
```
$ python level_from_archive.py --not_cuda --experiment_id {EXPERIMENT ID}
```

## Results

Results from our experiments can be found in the results folder.
