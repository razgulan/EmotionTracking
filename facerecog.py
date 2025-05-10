import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Arial']  # Destekleyen bir font
plt.rcParams['axes.unicode_minus'] = False


# Sınıf adları (veri klasör sırasına göre)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Görüntü parametreleri
IMG_SIZE = 48
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

# CLAHE için ayarlar (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_image(image_path):
    # Görüntüyü gri tonlamada oku
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Yeniden boyutlandır
    img = cv2.resize(img, IMG_SHAPE)

    # Kontrast iyileştirme uygula
    img = clahe.apply(img)

    # Normalize et (0-1 aralığına getir)
    img = img.astype('float32') / 255.0

    # CNN'lerde kanal boyutu için şekli (48, 48, 1) yap
    img = np.expand_dims(img, axis=-1)

    return img


def load_dataset(folder_path):
    data = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # angry, disgust, ...

    for label_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            try:
                img = preprocess_image(img_path)
                data.append(img)
                labels.append(label_index)
            except:
                print(f"Hata oluştu: {img_path}")

    return np.array(data), to_categorical(labels)


# Datasetleri yükle
train_dir = "Veriseti/train"
test_dir = "Veriseti/test"

X_train, y_train = load_dataset(train_dir)
X_test, y_test = load_dataset(test_dir)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")

# One-hot label'ları düz label'a çevir
y_train_labels = np.argmax(y_train, axis=1)

# Görüntüleri flatten et
X_train_flat = X_train.reshape((X_train.shape[0], -1))

# SMOTE uygulaması
smote = SMOTE(random_state=42)
X_smote_flat, y_smote_labels = smote.fit_resample(X_train_flat, y_train_labels)

# Geri dönüştür
X_smote = X_smote_flat.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
y_smote = to_categorical(y_smote_labels)

print(f"SMOTE sonrası: {X_smote.shape}, {y_smote.shape}")

X_train = X_smote
y_train = y_smote

# Eğitim verisi için augmentasyon CNN
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=64)

def plot_sample_images(X, y, class_names):
    plt.figure(figsize=(10, 5))
    shown_classes = set()
    count = 0

    for i in range(len(X)):
        class_index = np.argmax(y[i])
        if class_index in shown_classes:
            continue
        plt.subplot(2, 4, count + 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(class_names[class_index])
        plt.axis('off')
        shown_classes.add(class_index)
        count += 1
        if count == len(class_names):
            break
    plt.suptitle("Her sınıftan örnek görseller")
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, class_names):
    counts = np.argmax(y, axis=1)
    sns.countplot(x=counts, palette=sns.color_palette('Set2', n_colors=len(class_names)))
    plt.title("Sınıf Dağılımı (Train)")
    plt.xlabel("Sınıf")
    plt.ylabel("Görüntü Sayısı")
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.show()


def plot_pixel_intensity_histogram(X):
    flattened = X.reshape(-1)
    plt.hist(flattened, bins=50, color='gray', alpha=0.8)
    plt.title("Tüm Görüntülerde Piksel Yoğunluğu Dağılımı")
    plt.xlabel("Piksel Yoğunluğu (0-1)")
    plt.ylabel("Frekans")
    plt.grid(True)
    plt.show()

# Eğitim verisini kullanarak gösteriyoruz
plot_sample_images(X_train, y_train, class_names)
# plot_sample_images(X_smote, y_smote, class_names)
plot_class_distribution(y_train, class_names)
# plot_class_distribution(y_smote, class_names)
plot_pixel_intensity_histogram(X_train)
# plot_pixel_intensity_histogram(X_smote)

# 🔁 Gri görüntüleri RGB formatına dönüştür (cv2 ile)
X_train_rgb = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) for img in X_train])
X_test_rgb = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) for img in X_test])

# Normalize yeniden yapılır
X_train_rgb = X_train_rgb.astype('float32') / 255.0
X_test_rgb = X_test_rgb.astype('float32') / 255.0

# 🔧 ResNet50 base model
resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=Input(shape=(48, 48, 3))
)

# 🔎 Global average pooling ile özellik çıkarımı
x = resnet_base.output
x = GlobalAveragePooling2D()(x)
resnet_model = Model(inputs=resnet_base.input, outputs=x)

# 🧠 Özellikleri çıkar
features_train = resnet_model.predict(X_train_rgb, verbose=1)
features_test = resnet_model.predict(X_test_rgb, verbose=1)

