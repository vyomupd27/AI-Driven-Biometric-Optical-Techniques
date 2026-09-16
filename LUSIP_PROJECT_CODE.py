#=================================IMPORTS==============================================
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#=================================PREPROCESSING==============================================
IMG_SIZE = 64

def preprocess_mace(img):
    """For MACE and SVM — with Hanning window"""
    resized = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_CUBIC)
    eq = cv.equalizeHist(resized)
    norm = eq / 255.0
    win = np.outer(np.hanning(IMG_SIZE), np.hanning(IMG_SIZE))
    return norm * win

def preprocess_corr(img):
    """For Cross Correlation — without Hanning window"""
    resized = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_CUBIC)
    eq = cv.equalizeHist(resized)
    norm = eq / 255.0
    return norm

# Two sets of images
train_images_mace, train_images_corr, train_labels = [], [], []
test_images_mace,  test_images_corr,  test_labels  = [], [], []

for number in range(1, 41):
    folder_path = 'dataset/s' + str(number)
    for num in range(1, 11):
        path = folder_path + '/' + str(num) + '.pgm'
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)

        mace_img = preprocess_mace(img)
        corr_img = preprocess_corr(img)

        if num <= 8:
            train_images_mace.append(mace_img)
            train_images_corr.append(corr_img)
            train_labels.append(number)
        else:
            test_images_mace.append(mace_img)
            test_images_corr.append(corr_img)
            test_labels.append(number)

print(f"Train: {len(train_images_mace)}, Test: {len(test_images_mace)}")
#==============================================================================================


#==============================MACE FILTER BUILDING========================================================
all_filters = {}

for person in range(1, 41):

    # Step 1 - Get FFT of this person's images
    person_fft = []
    for i in range(len(train_images_mace)):
        if train_labels[i] == person:
            fft = np.fft.fft2(train_images_mace[i])
            person_fft.append(fft)

    # Step 2 - Power spectrum and D
    power_spectrums = []
    for fft in person_fft:
        pwr_spect = np.abs(fft) ** 2
        power_spectrums.append(pwr_spect)
    avg_power_spectrum = np.mean(power_spectrums, axis=0).reshape(-1)

    # Step 3 - Build matrix X
    fft_matrix = np.stack(person_fft, axis=0)
    fft_matrix = fft_matrix.reshape(fft_matrix.shape[0], -1).T

    # Step 4 - vector d
    vector_d = np.ones(fft_matrix.shape[1])

    # Step 5 - Compute H
    D_inverse = 1/(avg_power_spectrum + 0.1)
    D_inv_X = D_inverse.reshape(-1,1) * fft_matrix
    X_conj_D_inv_X = fft_matrix.conj().T @ D_inv_X
    X_conj_D_inv_X_inv = np.linalg.inv(X_conj_D_inv_X)
    X_conj_D_inv_X_inv_d = X_conj_D_inv_X_inv @ vector_d
    result = fft_matrix @ X_conj_D_inv_X_inv_d
    H = D_inverse * result
    H = H.reshape(64, 64)
    # No H normalization — keeps true MACE response

    # Step 6 - Store filter
    all_filters[person] = H

print(f"Filters built: {len(all_filters)}")
#==============================================================================================


#==============================PSR FUNCTION========================================================
def compute_psr_scores(img_fft, all_filters, peak_guard=5, window=20):
    psr_scores, peak_values = {}, {}

    for person, H in all_filters.items():
        corr = np.fft.ifft2(img_fft * np.conj(H)).real
        corr = np.fft.fftshift(corr)
        corr_abs = np.abs(corr)

        peak = corr_abs.max()
        pr, pc = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)

        r0, r1 = max(0, pr - window), min(IMG_SIZE, pr + window + 1)
        c0, c1 = max(0, pc - window), min(IMG_SIZE, pc + window + 1)
        region = corr_abs[r0:r1, c0:c1].copy()

        lr, lc = pr - r0, pc - c0
        region[max(0, lr - peak_guard):lr + peak_guard + 1,
               max(0, lc - peak_guard):lc + peak_guard + 1] = np.nan

        sidelobe = region[~np.isnan(region)]
        mean = sidelobe.mean()
        std  = sidelobe.std() + 1e-10

        psr_scores[person]  = (peak - mean) / std
        peak_values[person] = peak

    return psr_scores, peak_values
#==============================================================================================


#=======================================CROSS CORRELATION=================================================
phase_templates = {}
#Training
for person in range(1,41):
    templates = []
    for j in range(len(train_images_corr)):
        if train_labels[j] == person:
            phase_fft = np.fft.fft2(train_images_corr[j])
            phase_norm = phase_fft/(np.abs(phase_fft) + 1e-10)
            templates.append(phase_norm)

    phase_templates[person] = np.mean(templates, axis=0)

print(f"Phase templates built: {len(phase_templates)}")

#Testing
correct_cross = 0

for i in range(len(test_images_corr)):
    test_phase_fft = np.fft.fft2(test_images_corr[i])
    test_phase_norm = test_phase_fft/(np.abs(test_phase_fft) + 1e-10)

    best_score = 0
    best_person = 0

    for person in range(1, 41):
        # Use single averaged template
        train_norm = phase_templates[person]
        correlation = np.fft.ifft2(test_phase_norm * np.conj(train_norm)).real
        peak = np.max(np.abs(correlation))

        if peak > best_score:
            best_score = peak
            best_person = person

    if best_person == test_labels[i]:
        correct_cross += 1

cross_accuracy = correct_cross / len(test_images_corr) * 100
print(f"Phase Only Correlation Accuracy: {cross_accuracy}%")

#====================================================================================================


#==============================PSR-NORMALIZED SURFACE (for visualization)=====================
def psr_surface(corr_abs, peak_guard=5):
    
    peak = corr_abs.max()
   
    pr, pc = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    mask = np.ones_like(corr_abs, dtype=bool)
    r0, r1 = max(0, pr - peak_guard), min(corr_abs.shape[0], pr + peak_guard + 1)
    c0, c1 = max(0, pc - peak_guard), min(corr_abs.shape[1], pc + peak_guard + 1)
    mask[r0:r1, c0:c1] = False

    mean = corr_abs[mask].mean()
    std  = corr_abs[mask].std() + 1e-10

    return (corr_abs - mean) / std
#==============================================================================================


#=====================================MACE ONLY ACCURACY==================================================
correct = 0
for i in range(len(test_images_mace)):
    test_fft = np.fft.fft2(test_images_mace[i])
    psr_scores, _ = compute_psr_scores(test_fft, all_filters)
    predicted = max(psr_scores, key=psr_scores.get)
    if predicted == test_labels[i]:
        correct += 1

mace_accuracy = correct / len(test_images_mace) * 100
print(f"MACE Only Accuracy: {mace_accuracy}%")
#==============================================================================================


#=============================BUILD FEATURES FOR SVM================================================
train_features = []
for i in range(len(train_images_mace)):
    train_fft = np.fft.fft2(train_images_mace[i])
    psr_scores, peak_values = compute_psr_scores(train_fft, all_filters)
    combined = list(psr_scores.values()) + list(peak_values.values())
    train_features.append(combined)

train_features = np.array(train_features)
print(f"Train features shape: {train_features.shape}")

test_features = []
for i in range(len(test_images_mace)):
    test_fft = np.fft.fft2(test_images_mace[i])
    psr_scores, peak_values = compute_psr_scores(test_fft, all_filters)
    combined = list(psr_scores.values()) + list(peak_values.values())
    test_features.append(combined)

test_features = np.array(test_features)
print(f"Test features shape: {test_features.shape}")
#==============================================================================================


#========================================SVM==============================================
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
svm.fit(train_features, train_labels)
predictions = svm.predict(test_features)
svm_accuracy = accuracy_score(test_labels, predictions)
print(f"MACE + SVM Accuracy: {svm_accuracy * 100:.2f}%")
#==============================================================================================


#====================================ACCURACY COMPARISON==============================================
methods = ['Cross Correlation', 'MACE Only', 'MACE + SVM']
accuracies = [cross_accuracy, mace_accuracy, svm_accuracy * 100]

plt.figure(figsize=(8, 5))
plt.bar(methods, accuracies, color=['#2ecc71', '#e74c3c', '#3498db'], width=0.4)
plt.xlabel("Methods")
plt.ylabel("Accuracy %")
plt.title("Accuracy Comparison")
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.show()
#===============================================================================================


#===================================FAR AND FRR==================================================
genuine_scores = []
imposter_scores = []

for i in range(len(test_images_mace)):
    test_score_fft = np.fft.fft2(test_images_mace[i])
    psr_scores_graph, _ = compute_psr_scores(test_score_fft, all_filters)
    true_person = test_labels[i]
    for person in range(1, 41):
        if person == true_person:
            genuine_scores.append(psr_scores_graph[person])
        else:
            imposter_scores.append(psr_scores_graph[person])

print(len(genuine_scores))
print(len(imposter_scores))

all_scores = genuine_scores + imposter_scores
all_labels = [1] * len(genuine_scores) + [0] * len(imposter_scores)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer_index = np.argmin(np.abs(fnr - fpr))
eer = (fpr[eer_index] + fnr[eer_index]) / 2
print(f"EER: {eer * 100:.2f}%")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('True Acceptance Rate (1-FRR)')
plt.title('ROC Curve - MACE Filter')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(thresholds, fpr, label="FAR")
plt.plot(thresholds, fnr, label="FRR")
plt.axvline(x=thresholds[eer_index], color='r', linestyle='--', label=f'EER = {eer*100:.2f}%')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('FAR vs FRR - EER Analysis')
plt.legend()
plt.grid(True)
plt.show()
#====================================================================================================


#===================================MACE CORRELATION PEAK WITH PSR ENHANCEMENT=======================
# Pick a genuine person that exists in the test set
target_person = 5
imposter_person = 10

test_img = None
for i in range(len(test_labels)):
    if test_labels[i] == target_person:
        test_img = test_images_mace[i]
        break

test_fft = np.fft.fft2(test_img)

# Genuine: person test image vs SAME person's filter  -> sharp spike
genuine_corr = np.fft.fftshift(np.abs(
    np.fft.ifft2(test_fft * np.conj(all_filters[target_person]))))

# Imposter: same test image vs DIFFERENT person's filter -> flat, no peak
imposter_corr = np.fft.fftshift(np.abs(
    np.fft.ifft2(test_fft * np.conj(all_filters[imposter_person]))))

# Convert to PSR-scaled surfaces (real spike + real noise floor)
genuine_psr  = psr_surface(genuine_corr)
imposter_psr = psr_surface(imposter_corr)

# Meshgrid
x = np.arange(IMG_SIZE)
y = np.arange(IMG_SIZE)
X, Y = np.meshgrid(x, y)

# Share the same z-limit so the comparison is honest
zmax = max(genuine_psr.max(), imposter_psr.max()) * 1.05

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, genuine_psr, cmap='jet',
                 rstride=1, cstride=1, linewidth=0, antialiased=False)
ax1.set_title(f'Authentic (Person {target_person}) — Sharp Peak')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('PSR')
ax1.set_zlim(genuine_psr.min(), zmax)
ax1.view_init(elev=25, azim=45)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, imposter_psr, cmap='jet',
                 rstride=1, cstride=1, linewidth=0, antialiased=False)
ax2.set_title(f'Imposter (Person {imposter_person}) — No Peak')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('PSR')
ax2.set_zlim(genuine_psr.min(), zmax)   # same scale as genuine
ax2.view_init(elev=25, azim=45)

plt.suptitle('MACE Correlation Output (PSR-scaled)', fontsize=14)
plt.tight_layout()
plt.show()
#====================================================================================================