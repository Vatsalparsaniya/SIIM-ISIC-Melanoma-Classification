## Melanoma Hair Remove

 
* There are many images with body hair covering the lesion so, hair remove operation can be useful for model focus on lesion part.
*  Method for hair remove using CV2 
   
[Notebook](https://www.kaggle.com/vatsalparsaniya/melanoma-hair-remove)

[Discussion](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165582)    
 
```python
def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image
```

* this technique works on black-Hair only (Not white hair)

## Example of body hair remove with this method:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2907842%2F167afd50ab11911426494c40b0dee656%2F4.png?generation=1594370870226580&amp;alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2907842%2F03b23320df9ea9966efb09c2a60fc305%2F1.png?generation=1594370918408705&amp;alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2907842%2F556392470cd28765aeaff2e4055a831d%2F3.png?generation=1594370944655636&amp;alt=media)

## Visualizing all CV2 Operations 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2907842%2Fb156694a1947ec2798d05bf321883ec3%2Findex.png?generation=1594370566397745&amp;alt=media)


