## Single Model Output


#### Single model performace 2020 data  [M]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters |
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- | 
|1|EfficientNet**B6**<br>(noisy-student)|**512x512**|0.913|**0.9405**|❌|<details>Fold-1<br>max_auc=0.90<br>Fold-2<br>max_auc=0.90<br>Fold-3<br>max_auc=0.88<br></details>|<details> Fold=3<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|2|EfficientNet**B6**<br>(noisy-student)|**384x384**|0.913|**0.9389**|❌|<details>Fold-1<br>max_auc=0.92<br>Fold-2<br>max_auc=0.90<br>Fold-3<br>max_auc=0.90<br>Fold-4<br>max_auc=0.88<br>Fold-5<br>max_auc=0.89</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 
|3|EfficientNet**B6**<br>(noisy-student)|**256x256**|0.908|**-**|❌|<details>cosine_schedule<br>Focal_loss<br>Fold-1<br>max_auc=0.918<br>Fold-2<br>max_auc=0.910<br>Fold-3<br>max_auc=0.900</details>|<details> Fold=3<br>epochs=20<br>TTA=15<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>|
|4|EfficientNet**B6**<br>(imagenet)|**192x192**|0.904|**0.9132**|❌|<details>Fold-1<br>max_auc=0.907<br>Fold-2<br>max_auc=0.912<br>Fold-3<br>max_auc=0.895</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|5|EfficientNet**B6**<br>(noisy-student)|**128x128**|0.895|**0.9253**|❌|<details>Fold-1<br>max_auc=0.890 <br>Fold-2<br>max_auc=0.905 <br>Fold-3<br>max_auc=0.890 </details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|6|EfficientNet**B5**<br>(noisy-student)|**512x512**| 0.919 |**-**|❌|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.921<br>Fold-2<br>max_auc=0.930<br>Fold-3<br>max_auc=0.906</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|7|EfficientNet**B5**<br>(noisy-student)|**384x384**|0.918|**-**|❌|<details>cosine_schedule<br>Focal_loss<br>Fold-1<br>max_auc=0.924<br>Fold-2<br>max_auc=0.923<br>Fold-3<br>max_auc=0.909</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 
|8|EfficientNet**B5**<br>(imagenet)|**256x256**| 0.909 |**0.9219**|❌|<details>cosine_schedule<br>Focal_loss<br>Fold-1<br>max_auc=0.915<br>Fold-2<br>max_auc=0.914<br>Fold-3<br>max_auc=0.910</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|9|EfficientNet**B5**<br>(noisy-student)|**192x192**| 0.908 |**0.9201**|❌|<details>cosine_schedule<br>Focal_loss<br>Fold-1<br>max_auc=0.928<br>Fold-2<br>max_auc=0.919<br>Fold-3<br>max_auc=0.908<br>Fold-4<br>max_auc=0.885<br>Fold-5<br>max_auc=0.903</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>|
|10|EfficientNet**B5**<br>(noisy-student)|**128x128**| 0.908 |**-**|❌|<details>lr_callback<br>Focal_loss<br>Fold-1<br>max_auc=0.92<br>Fold-2<br>max_auc=0.903<br>Fold-3<br>max_auc=0.915<br>Fold-4<br>max_auc=0.893<br>Fold-5<br>max_auc=0.907</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 
|11|EfficientNet**B4**<br>(noisy-student)|**512x512**| 0.913 |**-**|❌|<details>cosine_schedule<br>Focal_loss<br>Fold-1<br>max_auc=0.920<br>Fold-2<br>max_auc=0.918<br>Fold-3<br>max_auc=0.904</details>|<details> Fold=5<br>epochs=20<br>TTA=15<br>INC2019 = [0,0,0]<br>INC2018 = [0,0,0]</details>| 
|12|EfficientNet**B4**<br>(noisy-student)|**384x384**| 0.920 |**0.9327**|❌|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.921<br>Fold-2<br>max_auc=0.927<br>Fold-3<br>max_auc=0.929<br>Fold-4<br>max_auc=0.899<br>Fold-5<br>max_auc=0.924</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>|
|13|EfficientNet**B4**<br>(noisy-student)|**256x256**| 0.918 |**-**|❌|<details>Lr_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.920<br>Fold-2<br>max_auc=0.930<br>Fold-3<br>max_auc=0.935<br>Fold-4<br>max_auc=0.891<br>Fold-5<br>max_auc=0.917</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 
|14|EfficientNet**B4**<br>(noisy-student)|**192x192**| 0.915 |**-**|❌|<details>lr_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.928<br>Fold-2<br>max_auc=0.913<br>Fold-3<br>max_auc=0.925<br>Fold-4<br>max_auc=0.896<br>Fold-5<br>max_auc=0.916</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 
|15|EfficientNet**B4**<br>(noisy-student)|**128x128**| 0.907 |**-**|❌|<details>Lr_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.918<br>Fold-2<br>max_auc=0.912<br>Fold-3<br>max_auc=0.911<br>Fold-4<br>max_auc=0.893<br>Fold-5<br>max_auc=0.904</details>|<details> Fold=5<br>epochs=15<br>TTA=15<br>INC2019 = [0,0,0,0,0]<br>INC2018 = [0,0,0,0,0]</details>| 


#### Single model performace 2020 + 2018 data [R]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters | 
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- | 
|1|EfficientNet**B6**<br>(imagenet)|**512x512**| 0.930 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.925<br>Fold-2<br>max_auc=0.931<br>Fold-3<br>max_auc=0.936</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|2|EfficientNet**B6**<br>(imagenett)|**384x384**| 0.917 |**0.9417**|✔️|<details>Upsampling M1,M3,M4<br>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.910<br>Fold-2<br>max_auc=0.925<br>Fold-3<br>max_auc=0.918</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|3|EfficientNet**B6**<br>(imagenet)|**256x256**| 0.913 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.914<br>Fold-2<br>max_auc=0.914<br>Fold-3<br>max_auc=0.912</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|4|EfficientNet**B6**<br>(imagenet)|**192x192**| 0.905 |**0.9302**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.903<br>Fold-2<br>max_auc=0.915<br>Fold-3<br>max_auc=0.899</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B6**<br>(noisy-student)|**128x128**| 0.893 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.898<br>Fold-2<br>max_auc=0.907<br>Fold-3<br>max_auc=0.880</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B5**<br>(noisy-student)|**512x512**| 0.931 |**0.9418**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.928<br>Fold-2<br>max_auc=0.931<br>Fold-3<br>max_auc=0.936</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|7|EfficientNet**B5**<br>(noisy-student)|**384x384**| 0.915 |**0.9467**|✔️|<details>Upsampling M1,M3,M4<br>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.915<br>Fold-2<br>max_auc=0.925<br>Fold-3<br>max_auc=0.911</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|8|EfficientNet**B5**<br>(noisy-student)|**256x256**| 0.918 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.914<br>Fold-2<br>max_auc=0.922<br>Fold-3<br>max_auc=0.918</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|9|EfficientNet**B5**<br>(imagenet)|**192x192**| 0.906 |**0.9297**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.907<br>Fold-2<br>max_auc=0.912<br>Fold-3<br>max_auc=0.901</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|10|EfficientNet**B5**<br>(noisy-student)|**128x128**| 0.892 |**0.9194**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.888<br>Fold-2<br>max_auc=0.903<br>Fold-3<br>max_auc=0.886</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|11|EfficientNet**B4**<br>(noisy-student)|**512x512**| 0.920 |**0.9376**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.918<br>Fold-2<br>max_auc=0.927<br>Fold-3<br>max_auc=0.919</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|12|EfficientNet**B4**<br>(noisy-student)|**384x384**| 0.913 |**0.9423**|✔️|<details>Upsampling M1,M3,M4<br>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.917<br>Fold-2<br>max_auc=0.917 <br>Fold-3<br>max_auc=0.907</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|13|EfficientNet**B4**<br>(noisy-student)|**256x256**| 0.913 |**0.9347**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.908<br>Fold-2<br>max_auc=0.922<br>Fold-3<br>max_auc=0.917</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|14|EfficientNet**B4**<br>(noisy-student)|**192x192**| 0.905 |**0.9264**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.905<br>Fold-2<br>max_auc=0.911<br>Fold-3<br>max_auc=0.900</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|15|EfficientNet**B4**<br>(noisy-student)|**128x128**| 0.895 |**0.9187**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.892<br>Fold-2<br>max_auc=0.906<br>Fold-3<br>max_auc=0.886</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 



#### 384 Series [S]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters | 
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- |  
|1|EfficientNet**B6**<br>(noisy-student)|**384x384**| 0.925 |**0.9399**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.916<br>Fold-2<br>max_auc=0.935<br>Fold-3<br>max_auc=0.925</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|2|EfficientNet**B5**<br>(noisy-student)|**384x384**| 0.930 |**0.9434**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.92<br>Fold-2<br>max_auc=0.90<br>Fold-3<br></details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|3|EfficientNet**B4**<br>(noisy-student)|**384x384**| 0.921 |**0.9314**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.92<br>Fold-2<br>max_auc=0.924<br>Fold-3<br>max_auc=0.920</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|4|EfficientNet**B3**<br>(noisy-student)|**384x384**| 0.915 |**0.9312**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.917<br>Fold-2<br>max_auc=0.921<br>Fold-3<br>max_auc=0.910</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B2**<br>(noisy-student)|**384x384**| 0.915 |**0.9350**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.914<br>Fold-2<br>max_auc=0.924<br>Fold-3<br>max_auc=0.921</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B1**<br>(noisy-student)|**384x384**| 0.919 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.914<br>Fold-2<br>max_auc=0.924<br>Fold-3<br>max_auc=0.921</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|7|EfficientNet**B0**<br>(noisy-student)|**384x384**| 0.911 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.915<br>Fold-2<br>max_auc=0.918<br>Fold-3<br>max_auc=0.900</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 



#### 768 Series [P]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters | 
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- |  
|1|EfficientNet**B6**<br>(imagenet)|**768x768**| 0.927 |**0.9496**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|1_1|EfficientNet**B6**<br>(imagenet)|**768x768**| 0.918 |**0.9368**|✔️|<details>cosine_schedule<br>BinaryCrossentropy<br>Fold-1<br>max_auc=0.914<br>Fold-2<br>max_auc=0.929<br>Fold-3<br>max_auc=0.913</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|2|EfficientNet**B5**<br>(imagenet)|**768x768**| 0.934 |**0.9462**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.936<br>Fold-2<br>max_auc=0.929<br>Fold-3<br>max_auc=0.937</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|3|EfficientNet**B4**<br>(imagenet)|**768x768**| 0.929 |**0.9422**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.932<br>Fold-2<br>max_auc=0.929<br>Fold-3<br>max_auc=0.934</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|4|EfficientNet**B3**<br>(imagenet)|**768x768**| 0.926 |**0.9454**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.929<br>Fold-2<br>max_auc=0.935<br>Fold-3<br>max_auc=0.923</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B2**<br>(imagenet)|**768x768**| 0.924 |**0.9394**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.930<br>Fold-2<br>max_auc=0.919<br>Fold-3<br>max_auc=0.929</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B1**<br>(imagenet)|**768x768**| 0.927 |**0.9371**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.929<br>Fold-2<br>max_auc=0.933<br>Fold-3<br>max_auc=0.922</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|7|EfficientNet**B0**<br>(imagenet)|**768x768**| 0.923 |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.920<br>Fold-2<br>max_auc=0.929<br>Fold-3<br>max_auc=0.924</details>|<details> Fold=3<br>epochs=20<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 


#### B7 Series [E]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters |
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- |
|1|EfficientNet**B7**<br>(imagenet)|**768x768**| 0.937 |**0.9417**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.933<br>Fold-2<br>max_auc=0.947</details>|<details> Fold=2<br>epochs=20<br>TTA=20<br>INC2019 = [0,0]<br>INC2018 = [1,1]</details>| 
|2|EfficientNet**B7**<br>(imagenet)|**512x512**| 0.934 |**0.9453**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.932<br>Fold-2<br>max_auc=0.934<br>Fold-3<br>max_auc=0.938</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|3|EfficientNet**B7**<br>(imagenet)|**384x384**| 0.926 |**0.9424**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.921<br>Fold-2<br>max_auc=0.930<br>Fold-3<br>max_auc=0.929</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|4|EfficientNet**B7**<br>(imagenet)|**256x256**| 0.921 |**0.9370**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.920<br>Fold-2<br>max_auc=0.925<br>Fold-3<br>max_auc=0.919</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B7**<br>(imagenet)|**192x192**| 0.915 |**0.9310**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.912<br>Fold-2<br>max_auc=0.920<br>Fold-3<br>max_auc=0.913</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B7**<br>(imagenet)|**128x128**| 0.893 |**0.9191**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.895<br>Fold-2<br>max_auc=0.907<br>Fold-3<br>max_auc=0.884</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 



#### 1024 Series [H]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters |
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- |
|1|EfficientNet**B7**<br>(imagenet)|**1024x1024**| - |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|2|EfficientNet**B6**<br>(imagenet)|**1024x1024**| 0.931 |**0.9423**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.938<br>Fold-2<br>max_auc=0.931<br>Fold-3<br>max_auc=0.929</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|3|EfficientNet**B5**<br>(imagenet)|**1024x1024**| 0.932 |**0.9482**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.938<br>Fold-2<br>max_auc=0.931<br>Fold-3<br>max_auc=0.933</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|4|EfficientNet**B4**<br>(imagenet)|**1024x1024**| 0.927 |**0.9405**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.935<br>Fold-2<br>max_auc=0.925<br>Fold-3<br>max_auc=0.921</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B3**<br>(imagenet)|**1024x1024**| - |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|6|EfficientNet**B2**<br>(imagenet)|**1024x1024**| - |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B1**<br>(imagenet)|**1024x1024**| - |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|6|EfficientNet**B0**<br>(imagenet)|**1024x1024**| - |**-**|✔️|<details>cosine_schedule<br>Focal loss<br>Fold-1<br>max_auc=0.926<br>Fold-2<br>max_auc=0.926<br>Fold-3<br>max_auc=0.930</details>|<details> Fold=3<br>epochs=15<br>TTA=20<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 



#### New Seed [N]

|No| model |Image-Size| CV-Score | LB-Score |External Data| Details |parameters |
|:---:| :--- |:---: | :---: | :---: |:---:|:--- | :--- | 
|1|EfficientNet**B7**<br>(imagenet)|**512x512**| 0.9377 |**-**|✔️|<details>BCE<br>Focal loss</details>|<details> Fold=3<br>epochs=20<br>M3<br>CoutOut<br>TTA=25<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>|
|2|EfficientNet**B6**<br>(imagenet)|**512x512**| 0.9345 |**-**|✔️|<details>BCE<br>Focal loss</details>|<details> Fold=3<br>epochs=20<br>M3<br>CoutOut<br>TTA=25<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|3|EfficientNet**B5**<br>(imagenet)|**512x512**| 0.9393 |**-**|✔️|<details>BCE<br>Focal loss</details>|<details> Fold=3<br>epochs=20<br>M3<br>CoutOut<br>TTA=25<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|4|EfficientNet**B5**<br>(imagenet)|**512x512**| 0.9334 |**-**|✔️|<details>BCE<br>Focal loss</details>|<details> Fold=3<br>epochs=20<br>M3<br>CoutOut<br>TTA=25<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
|5|EfficientNet**B5**<br>(imagenet)|**768x768**| 0.9383 |**-**|✔️|<details>BCE<br>Focal loss</details>|<details> Fold=3<br>epochs=20<br>M3<br>CoutOut<br>TTA=25<br>INC2019 = [0,0,0]<br>INC2018 = [1,1,1]</details>| 
