# AMLS_II_assignment23-24

## Description

- Kaggle website: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
- 5-class classification problem
- "research" competition

## Prepare

- you can download the competiton dataset from https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification/data
- Because of computational resource limitation, student composed 2100 images into a new dataset, including 100 random CBB (category 0) images, 210 random CBSD (category 1) images, 230 random CGM images (category 2), 1310 random CMD images (category 3), and 250 random Healthy images (category 4).
- Put 2100 images into 'Datasets' folder.
- Create or install the environment according to the requirements below.

## Point

- you can run the main.py to run the whole project. And, you can add and remove "#" to change the method in main.py.
- Because of the size of the original dataset, even though it was scaled down, main.py still takes 3-6 hours to run depending on the method in the CPU situation.
- If you want to see the results, you can browse directly to the "some recorded diagrams" folder to get the results you want. OR, you can browse to the .ipynb file stored in the "A" folder to get the results you want.

## Documents

- main.py (run the project)
- README (Introduce the project)
- train.csv (include all image_id and label, which can be download from the competition)
- label_num_to_disease_map.json (show the fullname about each label, which can be download from the competition)
- some recorded diagram (The folder show the some results about models)
- plot (The folder uses to store the result diagram from the main.py)
- Datasets (The folder uses to store the image data)
- A (The folder uses to store the whole code about each model - ViT and ResNeXt)
- environment.yaml (The copy of student's environment)

## Packet required
To run the code in this project, the following packages are required:
- `scikit-learn`
- `torchvision`
- `matplotlib`
- `tqdm`
- `seaborn`
- `torch`
- `timm`
- `Pillow`
- `pandas`
- `numpy`
- `opencv-python`
- `albumentations`
OR
The required environment has been export to file "environment.yaml". Use the following conda instruction to finish the environment setting.
