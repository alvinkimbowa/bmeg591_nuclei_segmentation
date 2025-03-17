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


## Acknowledgment  
This project is adapted from the Kaggle [NuInsSeg](https://www.kaggle.com/datasets/ipateam/nuinsseg/code?datasetId=1911713) and github repo [NuInsSeg](https://github.com/masih4/NuInsSeg) by Amirreza Mahbod.
The original code is licensed under the MIT License.  

## Refrences
[1] Mahbod, A., Polak, C., Feldmann, K., Khan, R., Gelles, K., Dorffner, G., Woitek, R., Hatamikia, S., & Ellinger, I. (2024). NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images. Scientific Data, 11(1), 295. https://doi.org/10.1038/s41597-024-03117-2

