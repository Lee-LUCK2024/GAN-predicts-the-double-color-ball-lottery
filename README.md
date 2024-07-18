# GAN-predicts-the-double-color-ball-lottery

The results are for reference!!!

Source of Double Color Ball Lottery Data: https://data.17500.cn/ssq_asc.txt

File Structure:
```
.
GAN-predicts-the-double-color-ball-lottery/
│
├── Data-preprocessing/
│   ├── ssq_asc.txt
|   └── ssq_number_2_img.py
│
├── GAN/
│   ├── data/
│   │   ├── dataset/
│   │   │   ├── 1.jpg   
│   │   │   ├── ...   
│   │   │   └── 3141.jpg
│   ├── log/
│   │   ├── D_epoch50_ssq.pth
│   │   └── G_epoch50_ssq.pth
│   ├── Generator_Discriminator.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
```

How to use:
1. if you just want to see the prediction results, pelase run predict.py
2. if you want to update the dataset and train your own checkpoint:
   1)  update the ssq_asc.txt in Data-preprocessing, using this webside: https://data.17500.cn/ssq_asc.txt
   2)  run ssq_number_2_img.py, and put the results in dataset folder
   3)  run train.py to train your own checkpoint
   4)  modify [generator_log_path = ' '] in predict.py and run it
