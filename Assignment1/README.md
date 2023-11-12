This is the Code Repo for **AAPMT: AGI Assessment Through Prompt and Metric Transformer**.





## Dataset

Two datasets we used is [`AGIQA-3K`](https://github.com/lcysyzxdxc/AGIQA-3k-Database/tree/main)  and  [`AIGCIQA2023`](https://github.com/wangjiarui153/AIGCIQA2023), which is in  `assets/AGIQA_3K`  and `assets/AIGCIQA2023`.  

* AGIQA_3K: `assets/AGIQA_3K` includes all the images with  `.jpg` format, and `assets/AGIQA-3K.csv` includes the rating information of these images.
* AIGCIQA2023: `assets/AIGCIQA2023/Images` includes 2400 images with `.png` format, and `assets/AIGCIQA2023/AIGCIQA2023.csv` includes the rating information of these images.

## Model

We put the code for model structure under `ImageReward` directory, where we basically use the structure of [Image Reward](https://github.com/THUDM/ImageReward) as backbone model, while adding some new elements including  Metric Transformer.

Image Reward's model checkpoint should be at `checkpoint/ImageReward/ImageReward.pt`



## Working Procedure You Need To Know

For convenience of debugging, we seperate directories for training on  [`AGIQA-3K`](https://github.com/lcysyzxdxc/AGIQA-3k-Database/tree/main)  and  [`AIGCIQA2023`](https://github.com/wangjiarui153/AIGCIQA2023).

### Model Training

Take training on AGIQA-3K as an example.

1. Model parameters' configuration is at `train_on_AGIQA_3K/config`. Training options are included in `train_on_AGIQA_3K/config/options.py`.

2. Dataset preparation is done in `train_on_AGIQA_3K/custom_dataset.py`, where dataset is split into training set and test set while maintaining the "content isolation" property (images with same prompt are in the same set)

3. Code running scripts is included in `train_on_AGIQA_3K/scripts`, which helps solve import problems. 

* To train the model for text-image correspondence(alignment) task,  run the script at`train_on_AGIQA_3K/scripts/train_imgreward_for_alignment.sh`
* To train the model for quality assessing task, run the script at `train_on_AGIQA_3K/scripts/train_imgreward_for_quality.sh`

When training finished, there will be a new checkpoint file under  `train_on_AGIQA_3K/checkpoint`.

### Get Score

After training models,  we should firstly input the path to the checkpoint we just derived into `train_on_AGIQA_3K/ImgReward.py`.  Then run `train_on_AGIQA_3K/scripts/get_ImgReward_results.sh` to get the scores given by the trained model on corresponding dataset. This script will output a `filename.csv` file under the directory `assets/results/AGIQA_3K`. 

Here `filename` could be modified in file `train_on_AGIQA_3K/ImgReward.py`

### Validation

With this `filename.csv` file, we could calculate PLCC and SRCC score.

For example, to calculate the PLCC and SRCC score of text-image correspondence, just input the path of  `filename.csv`  to the `table_list`  in  `train_on_AGIQA_3K/validation/valid_alignment.py` , then run the script at `train_on_AGIQA_3K/scripts/valid_alignment.sh`. Results will show in `train_on_AGIQA_3K/validation/results`, with time prefix.

For  `AIGCIQA2023`, it basically shares the code structure above, which would be omitted here.

## Reproduction of the Results

<u>In some `.csv` file, you can found `used for ...` which might help you understand where this data is used.</u>

* **Table 1** could be found at `train_on_AGCIQA2023/config/options.py` and `train_on_AGIQA_3K/config/options.py`.

* The PLCC and SRCC score used in **Figure 1** could be found at 

  1. `train_on_AGIQA_3K/validation/results/valid_alignment.txt`
  2. `train_on_AGCIQA2023/validation/results/valid_alignment_epoch=50.txt`

  and the visualization code could be found at `visualization/visualize_align.py`. The data used for calculating PLCC and SRCC score is from:

  **AGIQA_3K**

  * `assets/results/AGIQA_3K/AGIQA_3K_ImgRward_raw.csv`, for raw Image Reward model
  * `assets/results/AGIQA_3K/AGIQA_3K_ImgRward_trained_for_align.csv` , for model trained for text-image correspondence task.
  * `assets/results/AGIQA_3K/AGIQA_3K_ImgRward_trained_for_quality.csv`, for model trained for assessing image quality task.

  **AIGCIQA2023**

  * `/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_raw.csv`, for raw Image Reward model.
  * `/data/husky/ImageReward/assets/results/AIGCIQA2023/ImgRward_trained_for_align_epoch50.csv` , for model trained for text-image correspondence task.

  The `.csv` file above is derived by:

  1.  train the model for specific task. (like `train_on_AGIQA_3K/train_imgreward_for_alignment.py`)
  2.  get the model score using `train_on_AGIQA_3K/ImgReward.py`

  **All the python file here should be runned by script file in order to avoid import error.**

* The PLCC and SRCC score used in **Figure 2** could be found at 

  1. `train_on_AGIQA_3K/validation/results/valid_quality.txt`
  2. `train_on_AGCIQA2023/validation/results/valid_quality_epoch=50.txt`

  How to derive PLCC and SRCC score is exactly the same with the procedure in Figure 1.

* **Table 2** 

  * We change the prompt we used in `train_on_AGCIQA2023/train_imgreward_for_quality.py` to get three different model. Prompt2's results are exactly the same with the one in Figure 2.

  * PLCC and SRCC score could be found at `train_on_AGCIQA2023/validation/results/valid_quality_epoch=50.txt`, where you could also find the path to the `.csv` file used to calculate score.

* **Table 3**

  * The model is trained by `train_on_AGCIQA2023/train_imgreward_for_authenticity.py`
  * PLCC and SRCC score could be found at `train_on_AGCIQA2023/validation/results/valid_authenticity_epoch=50.txt`

* **Table 4**

  * The procedure here is to load a checkpoint of a model trained for task 1(image quality assessment for example) and retrained it for another task(text-image correspondence for example). We used

    * `train_on_AGIQA_3K/train_imgreward_for_alignment.py`
    * `train_on_AGIQA_3K/train_imgreward_for_quality.py` 

    to do this.

  * PLCC and SRCC score could be found at 

    * `train_on_AGCIQA2023/validation/results/valid_alignment_epoch=50.txt`
    * `train_on_AGCIQA2023/validation/results/valid_quality_epoch=50.txt`

* **Table 5**

  * Except score for Metric Transformer, other scores have already been mentioned above.
  * PLCC and SRCC score could be found at 
    * `train_on_AGCIQA2023/validation/results/valid_alignment_epoch=50.txt`
    * `train_on_AGCIQA2023/validation/results/valid_quality_epoch=50.txt`
    * `train_on_AGCIQA2023/validation/results/valid_authenticity_epoch=50.txt`
  * `train_on_AGCIQA2023/train_imgreward_for_all.py` is used to train Metric Transformer. You could use `train_on_AGCIQA2023/scripts/train_imgreward_for_all.sh` to run it.
  * The path of `.csv` files used to get PLCC and SRCC score could be found inside the `.txt` files.
  * These `.csv` files are derived by:

    1.  train the model for specific task.
    2.  get the model score using `train_on_AGCIQA2023/ImgReward_for_all.py`

* **Figure 5**

  * Data is stored at `train_on_AGCIQA2023/checkpoint/epoch_loss_for_all_iqa_epoch50.pkl`, which is saved during running `train_on_AGCIQA2023/train_imgreward_for_all.py`.
  * Code for visualization is at `visualization/loss/plot_loss.py`

* **Table 6**

  * Seed = 42 is the case above in Figure 1 and 2.
  * Change the seed by modifying the parameter we input into `init_seed()` function at
    * `train_on_AGIQA_3K/train_imgreward_for_alignment.py`
    * `train_on_AGIQA_3K/train_imgreward_for_quality.py`
  * PLCC and SRCC score could be found at 
    * `train_on_AGIQA_3K/validation/results/valid_alignment_seed=100.txt`
    * `train_on_AGIQA_3K/validation/results/valid_alignment_seed=200.txt`
    * `train_on_AGIQA_3K/validation/results/valid_quality_seed=100.txt`
    * `train_on_AGIQA_3K/validation/results/valid_quality_seed=200.txt`

* **Figure 6**

  * We first run `train_on_AGCIQA2023/scripts/ImgReward_disentangled.sh` to get score table (`/assets/results/AIGCIQA2023/ImgRward_disentangled.csv`) of different metrics.
  * Then we run `train_on_AGCIQA2023/scripts/valid_disentangled.sh` to get PLCC and SRCC scores as well as the pie chart.

Other figures could be found at `visualization` dir.

******



**Due to time constraints, please forgive any unclear expressions. If you have any questions, please contact hbh001098hbh@sjtu.edu.cn, and I will be happy to answer them for you.**
