<center><h1>Face Mesh Classification using MediaPipe and Custom CNN</h1></center>

<center><h2>Overview</h2></center>

<p>This project focuses on classifying facial features by leveraging MediaPipe's face mesh capabilities. A custom convolutional neural network (CNN) was built and trained on landmark-based image data. The model was designed to be efficient and adaptable for real-time or batch face analysis tasks.</p>

<center><h2>Dataset Preparation</h2></center>

<p>I created a custom dataset by extracting face mesh landmarks using MediaPipe and rendering them onto 96x96 images. The dataset is organized in the following format:</p>

<pre>
face_mesh_dataset/
├── Training/
│   ├── Class1/
│   ├── Class2/
│   └── ...
├── Testing/
│   ├── Class1/
│   ├── Class2/
│   └── ...
</pre>

<p>Each image shows facial landmarks either as black points on white background (initial) or white points on black background (improved version).</p>

<center><h2>Dependencies</h2></center>

<p>Install required packages using pip:</p>

<pre>
pip install mediapipe opencv-python tensorflow numpy matplotlib scikit-learn
</pre>

<center><h2>How to Use</h2></center>

<ol>
<li><b>Mount Google Drive</b> (if using Colab):</li>
<pre>
from google.colab import drive
drive.mount('/content/drive')
</pre>

<li><b>Set the Dataset Path</b>:</li>
<pre>
train_folder = '/content/drive/MyDrive/face_mesh_dataset/Training'
test_folder = '/content/drive/MyDrive/face_mesh_dataset/Testing'
</pre>

<li><b>Preprocess the Data</b>:</li>
<pre>
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_folder,
    target_size=(96, 96),
    batch_size=32,
    class_mode='sparse'
)

test_generator = datagen.flow_from_directory(
    test_folder,
    target_size=(96, 96),
    batch_size=32,
    class_mode='sparse'
)
</pre>

<li><b>Model Architecture</b>:</li>
<pre>
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2DTranspose

model = Sequential()
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(96, 96, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2DTranspose(32, (3, 3)))

model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
</pre>

<li><b>Compile and Train</b>:</li>
<pre>
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=50, validation_data=test_generator)
</pre>

<li><b>Plot Accuracy and Loss</b>:</li>
<pre>
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
</pre>
</ol>

<center><h2>Key Insight</h2></center>

<p>Originally, I visualized landmarks as black dots on a white canvas. However, this setup caused the model to struggle with feature extraction due to how MaxPooling works—pooling large dark (zero) regions resulted in lost feature clarity.</p>

<p>Switching to white dots on a black canvas helped MaxPooling2D highlight key facial features better, thereby improving training efficiency and increasing test accuracy from 60% to 82%.</p>

<center><h2>Results</h2></center>

<ul>
  <li>Final Training Accuracy: ~82%</li>
  <li>Model: Custom CNN with Conv2D, MaxPooling, Dropout, Conv2DTranspose</li>
  <li>Optimizer: Adam</li>
  <li>Loss: Sparse Categorical Crossentropy</li>
</ul>

<center><h2>Contact</h2></center>

<p>Created by Joel Alex John<br>
For questions or collaborations, connect on <a href="https://www.linkedin.com/in/joelalexj/">LinkedIn</a></p>

