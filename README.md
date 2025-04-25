# SAM-NuInsSeg

To get started, clone the repo and install the requirements, preferrably in a virtual environment:

```
git clone https://github.com/alvinkimbowa/bmeg591_nuclei_segmentation.git
cd bmeg591_nuclei_segmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

Download the full dataset from https://zenodo.org/records/10518968.

Unzip it and you're ready to go!

For a quick dirty training of U-Net, run the command below. Updated the `data_dir` path accordingly

```
python train.py --data_dir </path/to/dataset>/NuInsSeg --epochs 125 --batch_size 4 --visualize_every
 20 --save_every 20
```

# Zero‐Shot Testing
You can either run the provided shell wrapper or call the Python script directly. Either way, you’ll need to edit the arguments to suit your setup.

Using the shell script
```
bash run_evaluation.sh \
  --data_dir      <path/to/your/data> \
  --batch_size    <batch_size> \
  --prompt_type   <grid|random|bounding_box> \
  --num_pos_points <num_positive_points> \
  --vis_dir       <path/to/vis/output> \
  --model_name    <basesam|medsam|hugesam> \
  --model_weights <path/to/checkpoint.pth>
 --data_dir : root folder of your dataset
```
which arguments are as follows: 

--batch_size : number of images per batch

--prompt_type : one of grid, random, bounding_box

--num_pos_points : how many positive points to sample per image

--vis_dir : where to save the output visualizations

--model_name : which SAM variant to use: basesam (Base SAM), medsam (MedSAM), or sam (HugeSAM)

--model_weights : path to the pretrained SAM weights (e.g. Models/MedSAM/bbx/sam_best.pth)

Direct Python invocation
```
python zeroshot_sam.py \
  --data_dir      $DATA_DIR \
  --batch_size    1 \
  --prompt_type   $PROMPT_TYPE \
  --num_pos_points $NUM_POS_POINTS \
  --vis_dir       $VIS_DIR \
  --model_name    $MODEL_NAME \
  --model_weights Models/MedSAM/bbx/sam_best.pth
```
# Fine‐Tuning (Training) SAM
Again, you can use the helper script or call the trainer directly.

Using the shell script
```
bash run_training.sh \
  --data_dir    <path/to/your/data> \
  --epochs      <num_epochs> \
  --batch_size  <batch_size> \
  --model_name  <basesam|medsam|hugesam> \
  --prompt_type <grid|random|bounding_box>
```
Direct Python invocation
```
python train_sam.py \
  --data_dir    /projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/NuInsSeg \
  --epochs      50 \
  --batch_size  1 \
  --model_name  basesam \
  --prompt_type random
```
Tip: Before running either script, double-check all paths and argument values to make sure they match your environment.


## Acknowledgment  
This project is adapted from the Kaggle [NuInsSeg](https://www.kaggle.com/datasets/ipateam/nuinsseg/code?datasetId=1911713) and github repo [NuInsSeg](https://github.com/masih4/NuInsSeg) by Amirreza Mahbod.
The original code is licensed under the MIT License.  

## Refrences
[1] Mahbod, A., Polak, C., Feldmann, K., Khan, R., Gelles, K., Dorffner, G., Woitek, R., Hatamikia, S., & Ellinger, I. (2024). NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images. Scientific Data, 11(1), 295. https://doi.org/10.1038/s41597-024-03117-2

