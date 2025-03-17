<hr>
<h1 align="center">
  CPN-Diff <br>
  <sub>Cell-Pulse Network Diffusion Model with Schedule-Free Sampling for Noise-Robust Modality Translation</sub>
</h1>
<div align="center">
 Anonymous Authors
<span></span>
   <br>
</div>
<hr>

<hr>


Official PyTorch implementation of **CPN-Diff**. Experiments demonstrate that our method performs effectively across two medical datasets and two thermal infrared visible light facial datasets.

<p align="center">
  <img src="figures/frame.png" alt="frame" style="width: 1200px; height: auto;">
</p>

## 🐹 Installation

This repository has been developed and tested with `CUDA 11.7` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.yaml
conda activate cpn
```

## 🐼 Prepare dataset
The default data set class GetDataset requires a specific folder structure for organizing the data set.
Modalities (such as `T1, T2, etc.`) should be stored in separate folders, while splits `(train, test, and optionally val)` should be arranged as subfolders containing `2D` images named `slice_0.png or .npy, slice_1.png or .npy`, and so on.
To utilize your custom data set class, implement your version in `dataset.py` by inheriting from the `SNNDataset` class.

```
<datasets>/
├── <modality_a>/
│   ├── train/
│   │   ├── slice_0.png or .npy
│   │   ├── slice_1.png or .npy
│   │   └── ...
│   ├── test/
│   │   ├── slice_0.png or .npy
│   │   └── ...
│   └── val/ (The file does not exist by default)
│       ├── slice_0.png or .npy
│       └── ...
├── <modality_b>/
│   ├── train/
│   ├── test/
│   └── val/ (The file does not exist by default)
├── ...
  
```

## 🙉 Training Model

Execute the following command to start or resume training.
Model checkpoints are stored in the `/checkpoints/$LOG` directory.
The script supports both `single-GPU` and `multi-GPU` training, with `single-GPU` mode being the default setting.

The example training code is as follows: 
```
python train_Model.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 2 \
  --max_epoch 120 \
  --lr 1.5e-4 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_SNN_logs
```

### Argument descriptions

| Argument                  | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| `--input_channels`        | Dimension of images.                                                                                  |
| `--source` and `--target` | Source Modality and Target Modality, e.g. 'T1', 'T2'. Should match the folder name for that modality. |
| `--batch_size`            | Train set batch size.                                                                                 |
| `--lr`                    | Learning rate.                                                                                        |
| `--max_epoch`             | Number of training epochs (default: 120).                                                             |
| `--input_path`            | Data set directory.                                                                                   |
| `--checkpoint_path`       | Model checkpoint path to resume training.                                                             |
| `--lambda_l1`             | [Optional] Composite ratio of loss items.                                                             |
| `--lambda_l2`             | [Optional] Composite ratio of loss items.                                                             |
| `--lambda_perceptual`     | [Optional] Composite ratio of loss items.                                                             |

## 🐧 Training CPN

Run the following command to start tuning.
The predicted images are saved under `/checkpoints/$LOG/generated_samples` directory.
By default, the script runs on a `single GPU`. 

```
python train_CPN.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 2 \
  --which_epoch 120 \
  --gpu_chose 0 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_CPN_logs
```

## 🐣 Testing CPN

Run the following command to start testing.
The predicted images are saved under `/checkpoints/$LOG/generated_samples` directory.
By default, the script runs on a `single GPU`. 

```
python test_CPN.py \
        --input_channels 1 \
        --source T1 \
        --target T2 \
        --batch_size 2 \
        --which_epoch 120 \
        --gpu_chose 0 \
        --input_path ./datasets/BrainTs20 \
        --checkpoint_path ./checkpoints/brats_1to2_CPN_logs
```

### Argument descriptions

Some arguments are common to both training and testing and are not listed here. For details on those arguments, please refer to the training section.

| Argument        | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `--batch_size`  | Test set batch size.                                                  |
| `--which_epoch` | Model checkpoint path.                                                |

## 🐸 Checkpoint

Refer to the testing section above to perform inference with the checkpoints. PSNR (dB), SSIM (%) and MAE are listed as mean ± std across the test set.

The paper is currently undergoing blind review.

| Dataset | Task      | PSNR         | SSIM         | MAE           | Checkpoint                   |
|---------|-----------|--------------|--------------|---------------|------------------------------|
| BRATS   | T1→T2     | 25.53 ± 2.08 | 91.92 ± 1.73 | 0.0275 ± 4.68 | -                            |
| BRATS   | T2→T1     | 24.52 ± 1.82 | 92.38 ± 1.16 | 0.0313 ± 4.09 | -                            |
| OASIS3  | T1→T2     | 23.05 ± 6.49 | 80.52 ± 5.84 | 0.0298 ± 9.82 | -                            |
| OASIS3  | T2→T1     | 23.59 ± 6.12 | 82.10 ± 4.63 | 0.0310 ± 1.01 | -                            |
| TFW     | VIS.→THE. | 21.08 ± 3.73 | 77.86 ± 2.63 | 0.0515 ± 1.80 | -                            |



## 🦊 Code

The code for the `test_CPN` is open, and the code for the `train_Model` and `train_CPN` will be made public shortly.

## 🐭 Citation

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```

```

<hr>
