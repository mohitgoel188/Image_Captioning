# Image Captioning

An Image captioning model extracts complete detail of individual object and their associated relationship from image. The system can automatically generate a neural sentence to describe the image. This problem is extremely important, as well as difficult because it connects two major artificial intelligence fields: computer vision and natural language processing. Also, enabling computers to describe the visual world will lead to a great number of possible applications, such as producing natural human robot interactions, early childhood education, information retrieval, and visually impaired assistance, and so on.

## Getting Started


* The [data_cleaning_captions.py]() and [data_cleaning_images.py]() create all basic files in `./Data/Processed` by doing necessory preprocessing.
* The [train.py]() does all model construction and training. It finally saves the trained model in `./Data/Saved Models` folder.
* The [predict.py]() predicts the caption of the given image.

### Prerequisites

Python 3.6.x is required with following modules with their dependencies:-
* `keras`    
    -- For constructing model,training,predicting
    <!-- * jupyter notebook                            
    -- For interactive feedback -->
* `numpy`                                       
    -- For numerical computation on arrays
* `cv2`  
    -- For image processing
* `joblib`  
    -- For storing and loading ML models
* `nltk`
  -- For caption preprocessing and finding BLEU score.

### Installing

Python 3 can be found at [this link](https://www.python.org/downloads/)
All further modules can be installed by passing: `pip install module_name` in the command shell (cmd, Windows Powershell, terminal, etc.) 
*Example:*
```
pip install keras
```

*For cv2 run following command:*

```
pip install opencv-python
```
