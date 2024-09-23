# Radargrams as Sequences: A Method for The Semantic Segmentation of Radar Sounder Data
This repository contains scratch code for the paper [Radargrams as Sequences](https://ieeexplore.ieee.org/document/10641860), which was presented at [IGARSS 2024](https://www.2024.ieeeigarss.org), Athens, Greece.

The code contains a propotype of the presented work and readers could find it helpful to obtain a grasp of the idea behind the paper.

If you wish to train the main model, double check the hard-coded paths of dataset and model and run:
```
python main.py
```
To run inference, use:
```
python test.py
```



If you use the code or find it helpful, please cite the following paper:
```
@INPROCEEDINGS{10641860,
  author={Corso, Jordy Dal and Bruzzone, Lorenzo},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Radargrams as Sequences: A Method for The Semantic Segmentation of Radar Sounder Data}, 
  year={2024},
  volume={},
  number={},
  pages={8179-8183},
  keywords={Representation learning;Radar remote sensing;Visualization;Semantic segmentation;Semantics;Object segmentation;Manuals;Semantic segmentation;Radar sounder;Sequence;Label  propagation;MCoRDS},
  doi={10.1109/IGARSS53475.2024.10641860}}
```

For further readings on sequential processing of radar sounder data, refer to:
```
@ARTICLE{10677400,
  author={Corso, Jordy Dal and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={An Approach to Semantic Segmentation of Radar Sounder Data Based on Unsupervised Random Walks and User-Guided Label Propagation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Radar;Semantic segmentation;Instruments;Feature extraction;Training;Measurement;Deep learning;Radar sounder;random walks;unsupervised learning;label propagation;MCoRDS;SHARAD},
  doi={10.1109/TGRS.2024.3458188}}
```
