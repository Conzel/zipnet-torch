# zipnet-torch

This repository contains the code for the pytorch implementation of Zipnet. This encompasses the whole python side of the project: exporting weights, training and defining models.

We will include some examples for script usages in this 
file. Help can also be gained by calling the scripts with the "--help" argument.

## Model training
Models can be trained with the `training.py` script. 
This script has a lot of parameters that can be passed on to training, thus the help should be consulted.
Example usage:

```
python3 training.py -d ~/datasets/tiny-imagenet-200 --epochs 10 -lr 1e-4 --batch-size 16 --save --patch-size 64 64 --num-samples 200 --model fp_gdn
```

### Special role of checkpoints

Resulting checkpoints are produced in the same folder as the script was run. We always save the model name, the lambda used to train and the checkpoint from the end of training (`checkpoint-model=<...>-lambda=<...>.pth.tar`) and the best model `checkpoint-model=<...>-lambda=<...>-best.pth.tar`). These checkpoints can be used for other scripts, used models are inferred automatically by the naming of the checkpoints. Thus, do not rename them.

When a checkpoint is expected, one can also pass a mock-checkpoint to load pretrained models from CompressAI. This follows the pattern of `pretrained-model=<...>-q=<...>`, with q being a quality parameter from 1 to 8. The only currently supported pretrained model is `fp_pretrained`, the fully factorized model (bmshj2018_factorized).



## Exporting weights
Trained models can be exported to a rust weight format. We also accept pretrained models.

Example usage:
```
python3 export_weights.py checkpoints/fp_gdn/checkpoint-model=fp_gdn-lambda=0.1-best.pth.tar --out weights.npz
```

or for a pretrained model:

```
python3 export_weights.py pretrained-model=fp_pretrained-q=5 --out weights.npz
```

## Plotting R/D curves
Here, we proceed in two steps: First we generate the results in JSON format, then we plot them.

### Generating results
To generate the results, use `generate_rd_curve_results.py`.

If using a pretrained model, omit the `q-=<...>` suffix, as we will iterate over all quality parameters to get an actual curve.

Results will end up in the given results folder, as `<model-name>.json`.

Example usage:
```
python3 eval.py --checkpoints pretrained-model=fp_pretrained --images ~/datasets/Kodak --save-folder results
```

or for a pretrained model:
```
python3 eval.py --checkpoints pretrained-model=fp_pretrained --images ~/datasets/Kodak --save-folder results
```

### Plotting
The results that were generated before can be plotted with `plot_rd_curve.py`. 

To plot multiple results into the same plot, just specify multiple arguments to `--results-file`.

Example usage:
```
python3 plot_rate_distortion.py --results-file results/jpeg.json results/fp_gdn.json results/fp_relu.json results/fp_gdn_bias.json results/fp_pretrained.json -m ms-ssim --output rd_ms_ssim.png
```

## main.py
This script was conceived as a mini-demo to see the compression performance of a model on a single image.

Reports the actual bitrates of the performed compression (through our implementation via constriction and through the compressai implementation) and shows the reconstructed images (unless the `--no-show` flag is passed).