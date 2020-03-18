# generateAndScoreFakeImages 

Simple program to get fake images from thispersondoesnotexists and to evaluate them with AVA and koncept512

## Install

```Bash
mkdir content
cd content
git clone https://github.com/subpic/ava-mlsp.git
git clone https://github.com/subpic/koniq.git
touch __init__.py
git clone https://github.com/subpic/kutils.git
wget -O ava-mlsp/models/irnv2_mlsp_wide_orig/model_best_weights.h5 https://www.dropbox.com/s/16k0vh1dn7ls0cd/model_best_weights.h5?dl=1&raw=1
mkdir models
mkdir models/KonCept512
cd models/KonCept512
wget https://osf.io/uznf8/download
mv download model_best_weights.h5
```

## Usage

```Bash
# To generate and get number_of_images from thispersondoesnotexists:
python generateAndScore.py --generate=number_of_images --output_images=output_folder
# To evaluate thx to koncept512 the images from folder data_input:
python generateAndScore.py --koncept_inputs=data_input
# To evaluate thx to ava the images from folder data_input:
python generateAndScore.py --ava_inputs=data_input
# To do everything:
python generateAndScore.py --generate=number_of_images --output_images=myimages --koncept_inputs=myimages --ava_inputs=myimages
```