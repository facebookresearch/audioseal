from audioseal import AudioSeal
import os
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def get_best_accuracy(labels, scores, thresholds):
    # Convert scores and labels to NumPy arrays
    scores = np.array(scores)
    labels = np.array(labels)
    # Initialize the best accuracy and best threshold
    best_accuracy = 0
    best_threshold = 0
    # Iterate over all thresholds in the ROC curve
    for threshold in thresholds:
        # Convert scores to predictions using the current threshold
        predictions = (scores >= threshold).astype(int)
        # Calculate the accuracy
        accuracy = np.mean(predictions == labels)
        # Update the best accuracy and best threshold if the current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_accuracy, best_threshold


def from_path(dataset_path):
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    labels = []
    scores = []
    for file in os.listdir(dataset_path):
        if len(labels) % 2000 == 0:
            print(len(labels))
        path = os.path.join(dataset_path, file)
        wav, sr = sf.read(path)
        wav_torch = torch.from_numpy(wav).float().view(1, 1, -1)[:, :, :sr]
        watermark = model.get_watermark(wav_torch, sr)
        wav_torch_wm = wav_torch + watermark
        score_wm, _ = detector.detect_watermark(wav_torch_wm, sr)
        score, _ = detector.detect_watermark(wav_torch, sr)
        labels.append(1)
        scores.append(score_wm)
        labels.append(0)
        scores.append(score)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    acc, thres = get_best_accuracy(labels, scores, thresholds)
    acc_tpr = tpr[np.where(thresholds == thres)[0][0]]
    acc_fpr = fpr[np.where(thresholds == thres)[0][0]]
    tpr_at_fpr_3 = tpr[np.where(fpr >= 1e-3)[0][0]-1]
    auc = roc_auc_score(labels, scores)
    return acc_fpr, acc_tpr, acc, auc, tpr_at_fpr_3

def from_real_fake_path(real_dataset_path, fake_dataset_path):
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    labels = []
    scores = []
    for file in os.listdir(real_dataset_path):
        if len(labels) % 2000 == 0:
            print(len(labels))
        path = os.path.join(real_dataset_path, file)
        wav, sr = sf.read(path)
        wav_torch = torch.from_numpy(wav).float().view(1, 1, -1)[:, :, :sr]
        score, _ = detector.detect_watermark(wav_torch, sr)
        labels.append(0)
        scores.append(score)
    for file in os.listdir(fake_dataset_path):
        if len(labels) % 2000 == 0:
            print(len(labels))
        path = os.path.join(fake_dataset_path, file)
        wav, sr = sf.read(path)
        wav_torch = torch.from_numpy(wav).float().view(1, 1, -1)[:, :, :sr]
        watermark = model.get_watermark(wav_torch, sr)
        wav_torch_wm = wav_torch + watermark
        score_wm, _ = detector.detect_watermark(wav_torch_wm, sr)
        labels.append(1)
        scores.append(score_wm)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    acc, thres = get_best_accuracy(labels, scores, thresholds)
    acc_tpr = tpr[np.where(thresholds == thres)[0][0]]
    acc_fpr = fpr[np.where(thresholds == thres)[0][0]]
    tpr_at_fpr_3 = tpr[np.where(fpr >= 1e-3)[0][0]-1]
    auc = roc_auc_score(labels, scores)
    return acc_fpr, acc_tpr, acc, auc, tpr_at_fpr_3


if __name__ == '__main__':
    path_fake='/Users/robinsr/Downloads/fake_audios'
    real_path='/Users/robinsr/Downloads/AUDIO_OUTPUTS'
    acc_fpr, acc_tpr, acc, auc, tpr_at_fpr_3 = from_real_fake_path(real_path, path_fake)
    print(acc_fpr, acc_tpr, acc, auc, tpr_at_fpr_3)
