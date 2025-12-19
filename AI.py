try:
    from google.colab import drive
except Exception:
    drive = None
import os
import tensorflow as tf
import numpy as np
import cv2

faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def loadImages(folder_path, image_size=224, limit=None):

    images = []

    for filename in os.listdir(folder_path)[:limit]:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(folder_path, filename)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    filepath, target_size=(image_size, image_size)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
            except Exception as e:
                print(f'Error loading {filename}: {e}')

    return np.array(images)


def cropFaceFromFrame(frame_bgr, image_size=224):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    nextfaces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(nextfaces) > 0:
        x, y, w, h = max(nextfaces, key=lambda f: f[2] * f[3])
        face = frame_bgr[y:y + h, x:x + w]
    else:
        face = frame_bgr
    face = cv2.resize(face, (image_size, image_size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def loadFaceCrop(image_path, image_size=224):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f'Unable to load image: {image_path}')
    return cropFaceFromFrame(frame, image_size=image_size)


def loadImagesFromDrive(folder_path, image_size=224, limit=None):
    images = []

    for filename in os.listdir(folder_path)[:limit]:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filepath = os.path.join(folder_path, filename)
            try:
                frame = cv2.imread(filepath)
                if frame is None:
                    print(f'Error loading {filename}: unable to read image')
                    continue
                cropped = cropFaceFromFrame(frame, image_size=image_size)
                images.append(cropped[0])
            except Exception as e:
                print(f'Error processing {filename}: {e}')

    return np.array(images)


def predictImage(model, image_path, threshold=0.5):
    input_data = loadFaceCrop(image_path, image_size=224)
    preds = model.predict(input_data)
    probFake = float(preds[0][0])
    label = 1 if probFake >= threshold else 0

    return probFake, label


def predictVideo(model, video_path, frameSkip=10, threshold=0.5, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Unable to open video: {video_path}')
    frameProbs = []
    frameIndex = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIndex % frameSkip == 0:
            input_data = cropFaceFromFrame(frame, image_size=224)
            preds = model.predict(input_data)
            probFake = float(preds[0][0])
            frameProbs.append(probFake)
            if max_frames is not None and len(frameProbs) >= max_frames:
                break
        frameIndex += 1
    cap.release()
    if not frameProbs:
        raise ValueError('No frames processed from video.')
    result = float(np.mean(frameProbs))
    label = 1 if result >= threshold else 0
    return result, label, frameProbs


def createDeepfakeDetector():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                              input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    if drive is not None:
        drive.mount('/content/drive')

    path = '/content/drive/My Drive/DF40'

    if os.path.exists(path):
        print('Contents:')
        for item in os.listdir(path):
            print(f'  - {item}')
    else:
        print(f'Dataset not found at {path}')

    model = createDeepfakeDetector()
    model.summary()

    trainRealPath = '/content/drive/My Drive/DF40/train/real'
    trainFakePath = '/content/drive/My Drive/DF40/train/fake'
    valRealPath = '/content/drive/My Drive/DF40/val/real'
    valFakePath = '/content/drive/My Drive/DF40/val/fake'

    real_train = loadImages(trainRealPath, limit=10)
    fake_train = loadImages(trainFakePath, limit=10)
    print(f'Real training images: {len(real_train)}')
    print(f'Fake training images: {len(fake_train)}')

    real_labels = np.zeros(len(real_train))
    fake_labels = np.ones(len(fake_train))

    X_train = np.concatenate([real_train, fake_train])
    y_train = np.concatenate([real_labels, fake_labels])

    print(f'\nTraining set size: {len(X_train)} images')
    print(f'Labels: {np.unique(y_train, return_counts=True)}')

    history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    model.summary()
