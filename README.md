# OCT Image Classification

## Background

http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017).

![Image](https://github.com/amourav/OCT-Image-Classification/blob/cleanCode3/pics/2018.Kermany.Identifying%20Medical%20Diagnoses%20and%20Treatable%20Diseases%20by%20Image-Based%20Deep%20Learning_fig2.jpg)
Figure 1. Representative Optical Coherence Tomography Images and the Workflow Diagram [Kermany et. al. 2018] http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

(A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.


### Dataset

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

For additional information: see http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5


## Project Overview

This project was created to improve on the results reported by [Kermany et. al](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) for the OCT image classification system.

The reported method consisted of the (InceptionV3)[https://arxiv.org/abs/1512.00567] network pretrained on the (ImageNet)[http://www.image-net.org/] dataset, then fine tuned on a set of target OCT images. To improve on these results, we trained several networks such as (VGG16)[https://arxiv.org/abs/1409.1556], (ResNet50)[https://arxiv.org/abs/1512.03385], (Xception)[https://arxiv.org/abs/1610.02357] and compared them to this baseline.

![Image](https://github.com/amourav/OCT-Image-Classification/blob/cleanCode3/pics/comparison.png)
Figure 2. Comparison of several ImageNet pretrained CNNs, fine tuned on OCT scans, and evaluated on the test set.

Following we trained a meta classifier that combines the outputs of these several pretrained CNNs for the best possible prediction. 

![Image](https://github.com/amourav/OCT-Image-Classification/blob/cleanCode3/pics/meta.png)

Our comparison identified averaging probabilities across all models as a simple yet accurate technique which improved above the performance of any individual model.

## Installation

Download 'python 3.6' and install the packages in `requerements.txt`.

## Usage

### Data

Download the data from (here)[https://data.mendeley.com/datasets/rscbjbr9sj/2].

### Training

From the root project directory run `train.py` with the following arguments:

`-iTrn` Path to the training data directory. Must contain 4 subdirectories (NORMAL,CNV,DME,DRUSEN) corresponding to the image labels. Simply use the path to the downloaded training data

`-iVal` Path to the val data directory (optional). This is used for stopping the model training at the lowest validation loss. If not provided, a portion (10%) of training data will be used for validation.

example:
```
python ./train.py -iTrn "./RawData/OCT2017/train" -iVal "./RawData/OCT2017/val"
```


This will automatically save the contents of the model to the `modelOutput` directory.

### Inference

To predict the class of a new OCT image, run `predict.py` with the following arguments

`-i` - Path to directory of new oct images. Images must be in the root folder.
`o` - Path to directory where output will be generated.
`m` - Path to trained model directory.

example:
```
python ./predict.py -i "./RawData/testDir" -o "./modelOutput/metaClf_2019-06-22" -m "./modelOutput/metaClf_2019-06-22"
```



## Authors

Andrei Mouraviev

Eric Chou


## Acknowledgements

Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

**Inspiration**

Automated methods to detect and classify human diseases from medical images.
