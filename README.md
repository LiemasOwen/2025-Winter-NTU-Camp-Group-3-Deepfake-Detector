# Deepfake Detector Project 
A Python-Based tool that utilizes Convolutional Neural Network to detech deepfaked images and videos.

**Credits**
- **Nando, Ken (Sivarit)** - Programming the model, researching how to train and make the model, extracting and finding datasets
- **Owen** - GUI Vibe Coding, AI Prompting for debugging
- **Su Ya, Carissa, Abra** - No major contributions in the code, but we would like to thank them for making the rest of this project possible!


**Features**
- Detects whether an input is Real or Fake
- Accuracy of 0.9++ during the 5th (last) epoch
- Simple GUI to allow user to input images or videos and view the result of the detection.

**Specifications**
- Language - Python 3.10.10
- Libraries - Tensorflow, OpenCV, NumPy
- Training Data - Approx. 2000 images. 2 classes, real/fake. Real photos taken from Kaggle's Real Human faces dataset, Deepfake photos taken from DF40 dataset
- Hardware - Tested on a laptop with Intel Core Ultra 7 155H, and Nvidia RTX 4050 laptop GPU.

**The Model**
- The AI model was heavily based on the Youtube Video []
- We based our model off of that video, and some assistance from Gemini in Google Colab in debugging and fixing code. 

**The GUI**
- The GUI serves purely as a simple way to showcase our AI model's results after training for proof-of-concept.
- In the interest of time, it was purely Vibecoded by GPT 5.2 X-High Reasoning, debugged using SWE-1 and Penguin Alpha, all of which used within the Windsurf IDE's Cascade feature. The AI was prompted purely by a groupmate.

**Disclaimer**
- This model was made for a project presentation, it is not 100% accurate as it was trained with a small dataset.
