import os
import librosa
import librosa.display
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mfcc_Dict = {}

directory ='/home/horacio/PycharmProjects/musicGenreClassifier/musicGenreSoundTracks/Converted/'

def extractMFCC(audioFile):
    y, sr = librosa.load(audioFile, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs



for genreFolder in os.listdir(directory):
    genrePath = os.path.join(directory, genreFolder)
    for album in os.listdir(genrePath):
        albumPath = os.path.join(genrePath, album)
        for file in os.listdir(albumPath):
            filePath = os.path.join(albumPath, file)
            if file.endswith('.wav'):
                #mfccs, sr = extractMFCC(filePath)
                #print(f"MFCCS for {file}: {mfccs.shape}")
                print(f"Processing file: {file}")
                #extractMFCC(filePath)
                #mfcc_Dict[file] = extractMFCC(filePath)
                print("Extracting MFCCs...")
                mfccs = extractMFCC(filePath)
                label = genreFolder
                mfcc_Dict[file] = {'MFCCs': mfccs, 'label': label}
                #mfcc_Dict[] = label

#audio file features are represented as a single vector
#or 1D array
preProcesseData = []
for file, label in mfcc_Dict.items():
    mfccs = label['MFCCs']
    flattenedMFCCs = mfccs.flatten()
    label = label['label']
    preProcesseData.append((flattenedMFCCs, label))
    print(f"File: {file}, MFCCs: {mfccs.shape}, Genre: {label}")

print("\nIf th code runs up to here I can go to sleep")
#X is for the flattened MFCCs
X = []
#Y is used to transform the genre labels into numeric values
Y = []
#ML algo needs the label to be in numeric format
#LabelEncoder encodes the genre labels into numeric values
for features, label in preProcesseData:
    X.append(features)
    Y.append(label)
    labelEncoder = LabelEncoder()
    Y_Encoded = labelEncoder.fit_transform(Y)

X = np.hstack(X)
Y_Encoded = np.array(Y_Encoded)

#Use np.hstack() to convert the MFCC sequence before before appending
#So far I've stored the MFCCs and transformed them into a 1D array with their corresponding label
# Split the data into training and testing sets
num_samples = len(preProcesseData)
num_features = mfccs.shape[1]
X_reshaped = X.reshape(num_samples,num_features)
print("Beginning data splitting...")
X_train, X_test, Y_train_Encoded, Y_test_Encoded = train_test_split(X_reshaped, Y_Encoded, test_size=0.2, random_state=42)

print("Data has been split into training and testing sets.")
print("Number of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])

#If the code is reaches upto here I'll cry
#fuck
print("Data is ready for ML model training, congrats!")
#ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (176,) + inhomogeneous part.