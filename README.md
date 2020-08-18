# SIIM-ISIC-Melanoma-Classification


|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters | Link |
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- | :--- | 
|1|EfficientNet**B6**<br>(noisy-student)|**512x512**|0.913|**0.9405**|:x:|<details><summary></summary><ul><li>Fold-1<br>max_auc=0.90</li><li>Fold-2<br>max_auc=0.90</li><li>Fold-3<br>max_auc=0.88</li></ul></details>|<details><summary></summary><pre>Fold=3<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</pre></details>| [Click here](https://www.kaggle.com/vatsalparsaniya/triple-stratified-kfold-with-tfrecords?scriptVersionId=39012746)|
|2|EfficientNet**B6**<br>(imagenet)|**384x384**|0.913|**0.9389**|:heavy_check_mark:|<details><summary></summary><ul><li>Fold-1<br>max_auc=0.92</li><li>Fold-2<br>max_auc=0.90</li><li>Fold-3<br>max_auc=0.90</li><li>Fold-4<br>max_auc=0.88</li><li>Fold-5<br>max_auc=0.89</li></ul></details>|<details><summary></summary><pre>Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</pre></details>| [Click here](https://www.kaggle.com/vatsalparsaniya/triple-stratified-kfold-with-tfrecords/notebook?scriptVersionId=38953686)|
