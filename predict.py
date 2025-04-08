import os
import pickle
import numpy as np
import librosa

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_features(audio_path, sr=16000, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract audio features for grammar scoring
    """
    try:
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        features = {}
        
        # Spectral Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec)
        features['mel_spec_mean'] = np.mean(log_mel_spec, axis=1)
        features['mel_spec_std'] = np.std(log_mel_spec, axis=1)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features['contrast_mean'] = np.mean(contrast, axis=1)
        features['contrast_std'] = np.std(contrast, axis=1)
        
        # Prosodic Features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        pitch_mean = []
        for i in range(pitches.shape[1]):
            pitches_frame = pitches[:, i]
            mag_frame = magnitudes[:, i]
            if np.sum(mag_frame) > 0:
                pitch_mean.append(np.sum(pitches_frame * mag_frame) / np.sum(mag_frame))
            else:
                pitch_mean.append(0)
        
        features['pitch_mean'] = np.mean(pitch_mean)
        features['pitch_std'] = np.std(pitch_mean)
        
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Temporal Features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        features['speech_rate'] = np.sum(librosa.zero_crossings(y)) / (len(y) / sr)
        
        # Voice quality features
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        
        # Additional statistics
        features['duration'] = len(y) / sr
        
        non_silence = np.sum(np.abs(y) > 0.01) / len(y)
        features['non_silence_ratio'] = non_silence
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def predict_grammar_score(audio_path):
    """
    Predict grammar score for an audio file
    """
    # Extract features
    features = extract_features(audio_path)
    if not features:
        return None
    
    # Convert features to array format expected by the model
    feature_array = []
    for key in sorted(features.keys()):
        if isinstance(features[key], np.ndarray):
            feature_array.extend(features[key])
        else:
            feature_array.append(features[key])
    
    # Make prediction
    prediction = model.predict([feature_array])[0]
    
    # Ensure prediction is within valid range
    prediction = max(1.0, min(5.0, prediction))
    
    return prediction

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    score = predict_grammar_score(audio_path)
    
    if score is not None:
        print(f"Predicted grammar score: {score:.2f}")
    else:
        print("Could not predict grammar score for the provided audio file.")
