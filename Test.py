import cv2
from google.colab import files
import matplotlib.pyplot as plt

test_image_path = '/content/drive/My Drive/DF40/test/meme.jpg'

img = cv2.imread(test_image_path)

if img is None:
    print(f"Error: Could not load image from {test_image_path}. Please ensure the path is correct and Drive is mounted.")
else:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(imgRGB)
    plt.title(f"Image: {test_image_path.split('/')[-1]}")
    plt.axis('off')
    plt.show()

    probFake, label = predictImage(model, test_image_path, threshold=0.5)

    print(f"Fake probability: {probFake:.4f}")
    print("Prediction:", "FAKE" if label == 1 else "REAL")