# Fish-Recognition-androidApp
Fish Image Recognition in Android App 

In this project, we are going to develop an android app that uses mobile phone camera to classify 15 different types of fishes realtime.

## Data Collection and Machine Learning Pipeline

There isn’t a comprehensive labeled dataset available for fish species so I built my own dataset of 120+ images from scratch. 
I collected images using a few different methods outlined below.

### Data Sources:

1) Bing image search
2) instagram by hashtag (i.e. #browntrout)

After collecting the images, I manually reviewed all of the photos to ensure that the images matched their respective species label and removed any miscellaneous images that didn’t belong (fishing equipment, a river, tackle box, incorrectly labeled fish species, etc).
The data collection and cleaning phase took about one weeks. Now equipped with 120+ images across 15 fish species, I had an initial image dataset to start building a deep learning fish identification model. 
Although there were so many images that I ignore them because data collection is a time-consuming process.

### Building the Deep Learning Model
Rather than creating a CNN from scratch, we’ll use a pre-trained model and perform transfer learning to customize this model with our new dataset. The pre-trained model we’re going to use is MobileNet, and we'll fine tune on our images.

### Develop Android App
After we’ve built our TensorFlow model with transfer learning, we’ll use the TFLite converter to create a mobile-ready model variant. The model will then be used in an Android application that recognizes images captured by the camera realtime.

## Results
After 100 epochs, the test accuracy is about 45 percent for 15 specious. to reach a better result, we need to increase our data number t0 500+. lack of data is a bottleneck in deep learning!
