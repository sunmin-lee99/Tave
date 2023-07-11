import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
from pydub import AudioSegment
import pandas as pd
from tqdm.auto import tqdm
import os
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
import librosa
import warnings
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score


warnings.filterwarnings(action='ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

audio_model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

CFG = {
    'SR':16000,
    'SEED':42,
    'BATCH_SIZE':16,
    'TOTAL_BATCH_SIZE': 64,
    'EPOCHS':3,
    'LR':1e-4,
    'hidden_size': 1024,
}

train_df = pd.read_csv('./train6.csv')
test_df = pd.read_csv('./test.csv')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=CFG['SEED'])

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        feature.append(speech_array)
    return feature

train_x = speech_file_to_array_fn(train_df)
valid_x = speech_file_to_array_fn(valid_df)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        feature.append(speech_array)
    return feature

train_x = speech_file_to_array_fn(train_df)
valid_x = speech_file_to_array_fn(valid_df)

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, processor):
        self.x = x
        self.y = y
        self.processor = processor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_values = self.processor(self.x[idx], sampling_rate=CFG['SR'], return_tensors="pt", padding=True).input_values
        if self.y is not None:
            return input_values.squeeze(), self.y[idx]
        else:
            return input_values.squeeze()

def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence([torch.tensor(xi) for xi in x], batch_first=True)
    y = pad_sequence([torch.tensor([yi]) for yi in y], batch_first=True)  # Convert scalar targets to 1D tensors
    return x, y

def create_data_loader(dataset, batch_size, shuffle, collate_fn, num_workers=0):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=num_workers
                      )

train_dataset = CustomDataSet(train_x, train_df['label'], processor)
valid_dataset = CustomDataSet(valid_x, valid_df['label'], processor)

train_loader = create_data_loader(train_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)
valid_loader = create_data_loader(valid_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = audio_model
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, 8)
        self.linear = torch.nn.Linear(CFG['hidden_size'], 1024, bias=False)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output

    def predict_emotion(df):
        for path in tqdm(df['path']):
            # if not audio_file:
            # I fetched some samples with known emotions from here: https://www.fesliyanstudios.com/royalty-free-sound-effects-download/poeple-crying-252

            #   audio_file = audio_file
            sound = AudioSegment.from_file(path)
            sound = sound.set_frame_rate(16000)
            sound_array = np.array(sound.get_array_of_samples())
        # this model is VERY SLOW, so best to pass in small sections that contain
        # emotional words from the transcript. like 10s or less.
        # how to make sub-chunk  -- this was necessary even with very short audio files
        # test = torch.tensor(input.input_values.float()[:, :100000])

        input = processor(
            raw_speech=sound_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt")

        result = audio_model.forward(input.input_values.float())
        # making sense of the result
        id2label = {
            "0": "angry",
            "1": "fear",
            "2": "sad",
            "3": "disgust",
            "4": "neutral",
            "5": "happy"
        }
        interp = dict(zip(id2label.values(), list(round(float(i), 4) for i in result[0][0])))
        return interp


def validation(model, valid_loader, creterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iter(valid_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            output = model(x)
            loss = creterion(output, y)

            val_loss.append(loss.item())

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += predicted.eq(y).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy


def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
    model.to(device)
    creterion = nn.CrossEntropyLoss().to(device)

    best_model = None
    best_acc = 0

    for epoch in range(1, CFG['EPOCHS'] + 1):
        train_loss = []
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = creterion(output, y)
            loss.backward()

            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, creterion)

        if scheduler is not None:
            scheduler.step(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')

    print(f'best_acc:{best_acc:.5f}')
    metrics = best_acc
    return best_model

model = BaseModel()

optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

infer_model = train(model, train_loader, valid_loader, optimizer, scheduler)

test_df = pd.read_csv('./test.csv')

def collate_fn_test(batch):
    x = pad_sequence([torch.tensor(xi) for xi in batch], batch_first=True)
    return x

test_x = speech_file_to_array_fn(test_df)

test_dataset = CustomDataSet(test_x, y=None, processor=processor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_test)

def inference(model, test_loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in tqdm(iter(test_loader)):
            x = x.to(device)

            output = model(x)

            preds += output.argmax(-1).detach().cpu().numpy().tolist()

    return preds

preds = inference(infer_model, test_loader)

'''
confusion_emotions = ['angry', 'calm', 'disgust', 'fearful','happy','neutral','sad','surprised']
cm=metrics.confusion_matrix(valid_dataset ,np.argmax(,axis=-1))
df_cm=pd.DataFrame(cm,index=[i for i in confusion_emotions],columns=[i for i in confusion_emotions])
plt.figure(figsize=(10,7))
sn.heatmap(df_cm,annot=True)
'''

submission = pd.read_csv('./sample_submission.csv')
submission['label'] = preds
submission.to_csv('./epoch70_submission.csv', index=False)
