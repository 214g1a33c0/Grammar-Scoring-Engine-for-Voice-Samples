import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_DIR = '/home/ubuntu/grammar_scoring'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_AUDIO_DIR = os.path.join(DATASET_DIR, 'audios_train')
TEST_AUDIO_DIR = os.path.join(DATASET_DIR, 'audios_test')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

# Create directories if they don't exist
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load train/val split
train_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_split.csv'))
val_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'val_split.csv'))
test_data = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Function to extract audio features
def extract_features(audio_path, sr=16000, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract various audio features for grammar scoring
    
    Parameters:
    - audio_path: path to audio file
    - sr: sample rate
    - n_mfcc: number of MFCCs to extract
    - n_mels: number of Mel bands
    - n_fft: FFT window size
    - hop_length: hop length for frame-wise features
    
    Returns:
    - Dictionary of features
    """
    try:
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        features = {}
        
        # 1. Spectral Features
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec)
        features['mel_spec_mean'] = np.mean(log_mel_spec, axis=1)
        features['mel_spec_std'] = np.std(log_mel_spec, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features['contrast_mean'] = np.mean(contrast, axis=1)
        features['contrast_std'] = np.std(contrast, axis=1)
        
        # 2. Prosodic Features
        
        # Pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        pitch_mean = []
        pitch_std = []
        for i in range(pitches.shape[1]):
            pitches_frame = pitches[:, i]
            mag_frame = magnitudes[:, i]
            if np.sum(mag_frame) > 0:
                pitch_mean.append(np.sum(pitches_frame * mag_frame) / np.sum(mag_frame))
            else:
                pitch_mean.append(0)
        
        features['pitch_mean'] = np.mean(pitch_mean)
        features['pitch_std'] = np.std(pitch_mean)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 3. Temporal Features
        
        # Tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Speech rate approximation (using zero crossings)
        features['speech_rate'] = np.sum(librosa.zero_crossings(y)) / (len(y) / sr)
        
        # 4. Voice quality features
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        
        # 5. Additional statistics
        
        # Duration
        features['duration'] = len(y) / sr
        
        # Silence ratio
        non_silence = np.sum(np.abs(y) > 0.01) / len(y)
        features['non_silence_ratio'] = non_silence
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

# Extract features for a sample file to test
print("Testing feature extraction on a sample file...")
sample_file = os.path.join(TRAIN_AUDIO_DIR, train_data['filename'].iloc[0])
sample_features = extract_features(sample_file)
if sample_features:
    print(f"Successfully extracted {len(sample_features)} features")
    print("Feature names:", list(sample_features.keys()))

# Extract features for all training files
print("Extracting features for training files...")
train_features = []
train_labels = []

for idx, row in tqdm(train_data.iterrows(), total=len(train_data)):
    file_path = os.path.join(TRAIN_AUDIO_DIR, row['filename'])
    features = extract_features(file_path)
    if features:
        train_features.append(features)
        train_labels.append(row['label'])

# Extract features for validation files
print("Extracting features for validation files...")
val_features = []
val_labels = []

for idx, row in tqdm(val_data.iterrows(), total=len(val_data)):
    file_path = os.path.join(TRAIN_AUDIO_DIR, row['filename'])
    features = extract_features(file_path)
    if features:
        val_features.append(features)
        val_labels.append(row['label'])

# Extract features for test files
print("Extracting features for test files...")
test_features = []
test_filenames = []

for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
    file_path = os.path.join(TEST_AUDIO_DIR, row['filename'])
    features = extract_features(file_path)
    if features:
        test_features.append(features)
        test_filenames.append(row['filename'])

# Convert features to DataFrame
def features_to_df(features_list):
    # First, create a list of dictionaries
    feature_dicts = []
    for feature_dict in features_list:
        flat_dict = {}
        for key, value in feature_dict.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    flat_dict[f"{key}_{i}"] = v
            else:
                flat_dict[key] = value
        feature_dicts.append(flat_dict)
    
    # Convert to DataFrame
    return pd.DataFrame(feature_dicts)

# Convert to DataFrames
train_features_df = features_to_df(train_features)
val_features_df = features_to_df(val_features)
test_features_df = features_to_df(test_features)

print(f"Training features shape: {train_features_df.shape}")
print(f"Validation features shape: {val_features_df.shape}")
print(f"Test features shape: {test_features_df.shape}")

# Normalize features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_df)
val_features_scaled = scaler.transform(val_features_df)
test_features_scaled = scaler.transform(test_features_df)

# Save features and labels
np.save(os.path.join(FEATURES_DIR, 'train_features.npy'), train_features_scaled)
np.save(os.path.join(FEATURES_DIR, 'train_labels.npy'), np.array(train_labels))
np.save(os.path.join(FEATURES_DIR, 'val_features.npy'), val_features_scaled)
np.save(os.path.join(FEATURES_DIR, 'val_labels.npy'), np.array(val_labels))
np.save(os.path.join(FEATURES_DIR, 'test_features.npy'), test_features_scaled)

# Save feature column names and scaler
with open(os.path.join(FEATURES_DIR, 'feature_columns.pkl'), 'wb') as f:
    pickle.dump(train_features_df.columns.tolist(), f)

with open(os.path.join(FEATURES_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Save test filenames
with open(os.path.join(FEATURES_DIR, 'test_filenames.pkl'), 'wb') as f:
    pickle.dump(test_filenames, f)

print("Feature extraction completed successfully!")

# Plot feature importance (correlation with labels)
train_features_df['label'] = train_labels
correlations = train_features_df.corr()['label'].sort_values(ascending=False)
top_features = correlations.head(15)
bottom_features = correlations.tail(15)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
top_features.plot(kind='bar')
plt.title('Top 15 Features (Positive Correlation)')
plt.tight_layout()

plt.subplot(2, 1, 2)
bottom_features.plot(kind='bar')
plt.title('Bottom 15 Features (Negative Correlation)')
plt.tight_layout()

plt.savefig(os.path.join(FEATURES_DIR, 'feature_correlations.png'))
plt.close()

print("Feature correlation analysis completed and saved.")
