# Image Classification using MobileNetV2

## Description
A deep learning image classification project using transfer learning with MobileNetV2.

## Features
- Transfer learning
- Data augmentation
- Training/validation accuracy curves
- Confusion matrix
- Saved model

## Dataset
Use any image dataset organized like this:

```text
dataset/
  train/
    class1/
    class2/
  val/
    class1/
    class2/
  test/
    class1/
    class2/
```

Recommended datasets:
- PlantVillage
- Fruits 360
- Intel Image Classification

## Run

```bash
pip install -r requirements.txt
python train_image_classifier.py --data_dir dataset
```

## CV Description
Built an image classification model using CNN and transfer learning with MobileNetV2, including evaluation using accuracy and confusion matrix.
