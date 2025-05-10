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

print("SVM Başlıyor...")

def plot_svm_decision_boundary(svm, X, y, class_names):

    # SVM Karar Sınırını Görselleştirme

    # Özellikleri 2 boyuta indirgemek için PCA kullanıyoruz
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Sınıf etiketlerini iki boyuta indiriyoruz
    y_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y

    # Modeli yeniden eğitiyoruz (2D için)
    svm.fit(X_2d, y_labels)

    # Meshgrid (karar sınırları için)
    h = .02  # adım boyutu
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.get_cmap('Spectral'))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_labels, cmap=plt.get_cmap('Spectral'), edgecolors='k')

    # Sınıf isimlerini ekleyelim
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, loc="upper right")
    plt.title("SVM Karar Sınırları (2D PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    # Yalnızca PCA verisinin performansını ölçelim (2D SVM)
    y_preds_pca = svm.predict(X_2d)
    cm = confusion_matrix(y_labels, y_preds_pca)
    print("\n📊 Classification Report (2D PCA SVM Modeli):")
    print(classification_report(y_labels, y_preds_pca, target_names=class_names))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("📊 Confusion Matrix (2D PCA SVM Modeli)")
    plt.show()

# SVM modelimizi yeniden eğitiyoruz (2D PCA ile)
svm_decision = SVC(kernel='linear')

# Özellikleri (features_train ve features_test) ve etiketleri (y_smote) kullanıyoruz
plot_svm_decision_boundary(svm_decision, features_train, y_smote, class_names)


# Normal SVM (Yüksek Boyutlu)
svm_normal = SVC(kernel='linear')
svm_normal.fit(features_train, np.argmax(y_smote, axis=1))

# Test verisi ile tahmin
svm_preds_normal = svm_normal.predict(features_test)

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), svm_preds_normal)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("📊 Confusion Matrix (Normal SVM Modeli)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("\n📊 Classification Report (Normal SVM Modeli):")
print(classification_report(np.argmax(y_test, axis=1), svm_preds_normal, target_names=class_names))

print("SVM Bitiyor...")

# CNN modeli (Sequential)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("📊 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred, class_names):
    y_true_bin = label_binarize(np.argmax(y_true, axis=1), classes=list(range(len(class_names))))
    y_pred_bin = y_pred  # softmax çıkışı zaten olasılık

    plt.figure(figsize=(10, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title("ROC Eğrileri")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()



class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data

        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_val, axis=1)

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

        print(f"\n📊 Epoch {epoch + 1} Sonu:")
        print(f"🔹 Accuracy: {accuracy:.4f}")
        print(f"🔹 Precision: {precision:.4f}")
        print(f"🔹 Recall: {recall:.4f}")
        print(f"🔹 F1-Score: {f1:.4f}")



cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Derleme
cnn_model.compile(optimizer=Adamax(learning_rate=0.002),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Erken durdurma
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
metrics_callback = MetricsCallback(validation_data=(X_test, y_test))

# Eğitim
history = cnn_model.fit(
    #X_train, y_train,      yerine alt satır.
    train_generator,
    validation_data=(X_test, y_test),
    epochs=15,
    # batch_size=64,
    callbacks=[early_stop, metrics_callback],
    verbose=1
)

# Eğitim geçmişi grafiği
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



plot_training_history(history)

# Modelin tahminleri (CNN için)
cnn_preds = cnn_model.predict(X_test)

# Confusion Matrix
plot_confusion_matrix(y_test, cnn_preds, class_names)

# ROC Eğrisi
plot_roc_curve(y_test, cnn_preds, class_names)

y_pred_labels = np.argmax(cnn_preds, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print("\n📊 Classification Report (Final Sonuçlar):")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n📊 {model_name} Performans Değerleri:")
    print(f"🔹 Accuracy: {accuracy:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-Score: {f1:.4f}\n")

    return [model_name, accuracy, precision, recall, f1]


def plot_combined_confusion_matrices(y_true_svm, y_pred_svm, y_true_cnn, y_pred_cnn, class_names):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm_svm = confusion_matrix(y_true_svm, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title("SVM Confusion Matrix")
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")

    cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax[1])
    ax[1].set_title("CNN Confusion Matrix")
    ax[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.show()


def plot_combined_roc(y_true, y_pred_svm, y_pred_cnn, class_names):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
        fpr_svm, tpr_svm, _ = roc_curve(y_true_bin[:, i], y_pred_svm[:, i])
        fpr_cnn, tpr_cnn, _ = roc_curve(y_true_bin[:, i], y_pred_cnn[:, i])

        auc_svm = auc(fpr_svm, tpr_svm)
        auc_cnn = auc(fpr_cnn, tpr_cnn)

        plt.plot(fpr_svm, tpr_svm, label=f"SVM {class_names[i]} (AUC = {auc_svm:.2f})", linestyle="--")
        plt.plot(fpr_cnn, tpr_cnn, label=f"CNN {class_names[i]} (AUC = {auc_cnn:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title("SVM ve CNN - ROC Eğrileri")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


# 🔹 SVM Performans Analizi
y_true_svm = np.argmax(y_test, axis=1)
svm_preds_normal = svm_normal.predict(features_test)
svm_perf = evaluate_model(y_true_svm, svm_preds_normal, "SVM")

# 🔹 CNN Performans Analizi
y_true_cnn = np.argmax(y_test, axis=1)
y_pred_cnn = np.argmax(cnn_preds, axis=1)
cnn_perf = evaluate_model(y_true_cnn, y_pred_cnn, "CNN")

# 🔹 Performans Karşılaştırma Tablosu
performance_df = pd.DataFrame([svm_perf, cnn_perf], columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n📊 Performans Karşılaştırma Tablosu:")
print(performance_df)

# 🔹 Confusion Matrices
plot_combined_confusion_matrices(y_true_svm, svm_preds_normal, y_true_cnn, y_pred_cnn, class_names)

# 🔹 ROC Eğrileri
cnn_preds_prob = cnn_preds  # Softmax çıkışı
svm_preds_prob = svm_normal.decision_function(features_test)
plot_combined_roc(y_true_svm, svm_preds_prob, cnn_preds_prob, class_names)


##### Tahmin #####

def preprocess_test_image(image_path, img_size=48):
    """
    Dışardan yüklenen bir resmi uygun formata dönüştürür.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = clahe.apply(img)  # CLAHE uygulaması
    img = img.astype('float32') / 255.0  # Normalizasyon
    img_cnn = np.expand_dims(img, axis=-1)  # CNN için (48, 48, 1)

    # CNN için RGB formatına dönüştürme
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype('float32') / 255.0  # Normalizasyon
    img_rgb = np.expand_dims(img_rgb, axis=0)  # (1, 48, 48, 3)

    # SVM için flatten (özellik çıkarımı için)
    img_flat = np.expand_dims(img.flatten(), axis=0)  # (1, 48 * 48)

    return img_cnn, img_rgb, img_flat


def predict_image(image_path, svm_model, cnn_model, resnet_model, class_names):
    """
    Verilen bir resmi SVM ve CNN ile sınıflandırır.
    """
    print(f"\n📊 Test Resmi: {image_path}")

    # Görüntüyü ön işleme
    img_cnn, img_rgb, img_flat = preprocess_test_image(image_path)

    # CNN Tahmini
    cnn_pred_prob = cnn_model.predict(np.expand_dims(img_cnn, axis=0))
    cnn_pred_class = np.argmax(cnn_pred_prob)
    print(f"🔹 CNN Tahmini: {class_names[cnn_pred_class]} ({cnn_pred_prob[0][cnn_pred_class] * 100:.2f}%)")

    # ResNet ile özellik çıkarımı (SVM için)
    img_resnet_features = resnet_model.predict(img_rgb)

    # SVM Tahmini
    svm_pred_class = svm_model.predict(img_resnet_features)[0]
    print(f"🔹 SVM Tahmini: {class_names[svm_pred_class]}")

    # Görsel Gösterimi
    img_display = cv2.imread(image_path, cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f"CNN: {class_names[cnn_pred_class]} ({cnn_pred_prob[0][cnn_pred_class] * 100:.2f}%)\n"
              f"SVM: {class_names[svm_pred_class]}")
    plt.axis('off')
    plt.show()


# 📌 Kullanım:
image_path_happy = "HappyFemaleFace.jpg"
predict_image(image_path_happy, svm_normal, cnn_model, resnet_model, class_names)
image_path_disgusting = "DisgustingFemaleFace.jpg"
predict_image(image_path_disgusting, svm_normal, cnn_model, resnet_model, class_names)
image_path_sad = "SadFemaleFace.jpg"
predict_image(image_path_sad, svm_normal, cnn_model, resnet_model, class_names)
image_path_surprise = "SurprisedFemaleFace.jpg"
predict_image(image_path_surprise, svm_normal, cnn_model, resnet_model, class_names)