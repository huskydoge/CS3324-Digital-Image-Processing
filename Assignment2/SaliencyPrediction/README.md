# TranBlend: An Experimental Fusion of Textual Input in Saliency Prediction Models

This is code repo for `TranBlend: An Experimental Fusion of Textual Input in Saliency Prediction Models`

## Requirements
run `pip install -r requirements.txt` to install all the dependencies.

## Dataset
The dataset used in this report is [SJTU-TIS](https://pan.baidu.com/s/12DqeiOq_5taO4AkRdYOXUw?pwd=b2zu), and relevant paper is [here](https://ieeexplore.ieee.org/abstract/document/10182000).

## Code Structure

- `assets/saliency`: contains the dataset information.
  - `assets/saliency/fixation`: contains the eye fixation map. 
  - `assets/saliency/images`: contains the image data.
  - `assets/saliency/map`: contains the saliency map data (Ground Truth).
  - `assets/saliency/process_data.py`: preprocess the dataset and get those .csv files for convenience.
- `ImageReward`: contains the code for main model.
  - `ImageReward/TranSalNet`: contains the [TransalNet](https://github.com/ljovo/transalnet) Model.
    - `ImageReward/TranSalNet/pretrained_models`: contains the pretrained model for TransalNet. We use TranSalNet-Dense in this experiment, make sure you have downloaded the correct pretrained model.
    - `ImageReward/TranSalNet/FeatureBlendNet.py`: contains the Blending Block Module displayed in Fig 1 of the report.
    - `ImageReward/TranSalNet/TranSalNet_Dense.py`: contains the TranSalNet-Dense model structure. It's where we use the Blending Block Module to blend the textual input and visual input.
  - `ImageReward/ImageReward.py`: contains the BLIP model structure (acutally use the code from [here](https://github.com/THUDM/ImageReward)), and is where we use TranSalNet_Dense to cope with textual input and visual input.
  - `ImageReward/checkpoint/ImageReward`: contains the pretrained model for BLIP([ImageReward](https://pan.baidu.com/s/1WbrwEQbgfpWs0YaNjaiFwg?pwd=psqv)). Make sure you have downloaded the correct pretrained model and put it here.
- `train_on_saliency`: main dir for experiments
  - `train_on_saliency/config/option.py`: training options for the model.
  - `train_on_saliency/train_imgreward_for_saliency`: main file, training starter.
  - `train_on_saliency/custom_dataset`: data preparation, package `.csv` files at `assets/saliency` into python classes.
  - `train_on_salinecy/visualizatio_for_paper`: visualization for the report. 
    - `train_on_salinecy/visualizatio_for_paper/assets`: '.csv' files converted from tensorBoard event files, used for visualizations.
    - `train_on_salinecy/visualizatio_for_paper/colored_pred`: colored prediction in figure 1.
    - `train_on_salinecy/visualizatio_for_paper/transalNet_results`: visualization of figure 4.
    - `train_on_salinecy/visualizatio_for_paper/transalNet_blendNet_contrast`: visualization of figure 5.
    - `train_on_salinecy/visualizatio_for_paper/fixrate_ablation`: visualization of figure 6.
- `requirements.txt`: contains the dependencies of this project.

## Training
run `train_on_saliency/scripts/train.sh` to train the model.
```text
usage: train_imgreward_for_saliency.py

[-h] [--config CONFIG] [--seed SEED] [--savepath SAVEPATH] [--preload_path PRELOAD_PATH] [--rank_pair]                                   
[--model_name {transalnet,blend}] [--batch_size BATCH_SIZE] [--accumulation_steps ACCUMULATION_STEPS] [--epochs EPOCHS]       
[--task {all,pure,non_salient,whole,salient,oneforall}] [--train-iters TRAIN_ITERS] [--reshape RESHAPE] [--use_cross {0,1}]
[--fix_sal_encoder FSE] [--fix_sal_decoder FSD] [--distributed DISTRIBUTED] [--gpu_num GPU_NUM] [--gpu_id GPU_ID] [--device DEVICE]      
[--loss {mse,saloss}] [--cc_w CC_W] [--sim_w SIM_W] [--kldiv_w KLDIV_W] [--nss_w NSS_W] [--mse_w MSE_W] [--load_emb]                     
[--load_pair_store] [--fix_base] [--fix_rate FIX_RATE] [--lr LR] [--lr-decay-iters LR_DECAY_ITERS] [--train_iters TRAIN_ITERS]           
[--lr-decay-style {constant,linear,cosine,exponential,inverse_square_root}] [--lr-decay-ratio LR_DECAY_RATIO] [--warmup WARMUP]          
[--adam-beta1 ADAM_BETA1] [--adam-beta2 ADAM_BETA2] [--adam-eps ADAM_EPS] [--clear_visualizer] [--std_log]                               
[--valid_per_epoch VALID_PER_EPOCH] [--test_ckpt TEST_CKPT]
```
### Main Parameters
- `--epochs`: number of epochs to train. 
- `--batch_size`: batch size, default is `32` as used in the report.
- `--task`: five tasks to train, corresponding to five types of image text pair in SJTU-TIS. `oneforall` means simultaneously training on all type of tasks on SJTU-TIS dataset (training set, 80% of whole dataset)
- `--lr`: learning rate, default is `5e-6` as used in the report.
- `--loss`: loss function, default is `saloss` as used in the report.
- `--cc_w`: weight of correlation coefficient loss, default is `-2` as used in the report.
- `--sim_w`: weight of similarity loss, default is `-1` as used in the report.
- `--kldiv_w`: weight of KL divergence loss, default is `10` as used in the report.
- `--nss_w`: weight of NSS loss, default is `-1` as used in the report.
- `--mse_w`: weight of MSE loss, default is `4` as used in the report.
- `--fix_rate`: fix rate of ImageReward(BLIP) model, default is `0.9` as used in the report.
- `--model_name`: model name, default is `blend` as used in the report. Use `transalNet`for raw TranSalNet-Dense model.
- `--fix_sal_encoder`: fix rate of the encoder of TranSalNet-Dense model, default is `0.0` as used in the report.
- `--fix_sal_decoder`: fix rate of the decoder of TranSalNet-Dense model, default is `0.0` as used in the report.
- `--use_cross`: whether to use cross attention in TranSalNet-Dense model, default is `0` as used in the report.

### Steps

1.Warm up TranBlend model as used in the report, run the following command:
```bash
train_on_saliency/scripts/train.sh --model_name=blend --epochs=50 --task=oneforall --fse=0.0 --fse=0.0 --use_cross=0
```
2.Get the results of TranSalNet-Dense model in figure 4, run the following command:
```bash
train_on_saliency/scripts/train.sh --model_name=transalnet --epochs=25--task=task
```
here task could be chosen from `pure`, `non_salient`, `whole`, `salient`, `all` corresponding to the five types of image text pair in SJTU-TIS.
To get the results of TranBlend on those tasks is the same as above, just change the `model_name` to `blend`, and epochs to 15(due to limited time, we use epoch=15 in experiment, which is actually enough since there is no obvious increase in model's performance after).

3.Get MSE loss overfitting results in figure 2 and 3, run the following command:
```bash
train_on_saliency/scripts/train.sh --model_name=TransalNet --epochs=25 --task=all --loss=mse
```

## Visualization and Datas
All the results are loaded to BaiduYun. Here we collect all the training data (in TensorBoard event form) used in the report [here](https://pan.baidu.com/s/1ATTGdj2jJ0L_NUf1hjHACw?pwd=mi3d)
. Results have been well named, so it's easy to find the corresponding results for each figure in the report.
- `blend_<task>` is data of TranBlend used in figure 5 and table 2
- `transalnet_<task>` is data of TranSalNet-Dense used in figure 4,5 and table 2
- `transalnet_loss=mse_all` is data of TranSalNet-Dense used in figure 2
- `fse={:2f}_fse={:2f}_non-salient_blend` is data of TranBlend used in ablation study, figure 6

Note that using tensorBoard, you can open event file to see images derived during training and testing process, which could provide you with more information than the images in the report.

- Other visualization figures are in `train_on_saliency/visualization_for_paper/figures`
- Example used in figure 1 is in `train_on_saliency/salient/figure1` and `train_on_saliency/non_salient/figure1`.
- Example used in figure 7 is in `train_on_saliency/non_salient/figure7`.
