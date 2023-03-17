import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torchaudio
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
import json
from torch.utils.data import Dataset, DataLoader
save_dir="/model_dir"
def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)



def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:

        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

#加载数据
print("load数据开始")
def getFiles(data_dir,transition):
    content = []
    names = []
    data_list = os.listdir(data_dir)
    filenames = transition["filename"].tolist()
    text = transition["text"].tolist()
    for i in data_list:
        ab_path = data_dir + "/" +i
        # print(ab_path)
        files = os.listdir(ab_path)
        for j in files:
            tmp_file = j[:-4]

            if tmp_file in filenames:
                index = filenames.index(tmp_file)
                file_path = ab_path + "/" + j
                names.append(file_path)
                content.append(text[index])
    return names,content

#load magic data上的数据
def load_long_audio(data_dir,txt_dir=None,txt=False):
    wave_list = os.listdir(data_dir)
   
    txt_list = os.listdir(txt_dir)
    texts = []
    if txt:
        audio_file,text_file = [],[]
        for text in txt_list:
            file_catche = []
            with open(txt_dir + "/" + text) as fp:
                lines = fp.read().splitlines()
                for cont in lines:
                    sentence = cont.split("\t")[3]
                    sentence = sentence.replace('[SIL]',"<sos>").replace('[ENS]','<eos>').replace('[*]',"").replace(" ","")
                    if sentence not in ['<eos>','<sos>']:
                        sentence = " ".join([i for i in sentence])
                    texts.append(sentence)
        return wave_list,texts
        
    
    return wave_list

def load_audio_file(data_file):
    with open(data_file) as f:
        lines = f.read().splitlines()
    wav_file = [i.split(",")[0] for i in lines]
    txts = [" ".join(i.split(',')[1:]) for i in lines]
    return wav_file,txts
print("load first data")
data_path = "/workspace/data/audio/data_aishell/wav"
train_path = data_path +"/train"
dev_path = data_path +"/dev"
test_path = data_path +"/test"
transition = pd.read_csv("/workspace/data/audio/data_aishell/transcript/transition11.csv")
# filenames = transition["filename"].tolist()
# text = transition["text"].tolist()
transition['text'] = transition['text'].apply(lambda x: " ".join([i for i in x]))

train_name1,train_content1= getFiles(train_path,transition)
dev_name1,dev_content1 = getFiles(test_path,transition)

print("load second data")
data_path = "/workspace/data/audio/speech/waves"

train_name, train_texts = load_audio_file("{}/train.txt".format(data_path))
train_name = ["{}/train/{}".format(data_path,i) for i in train_name]

dev_name, dev_texts = load_audio_file("{}/dev.txt".format(data_path))
dev_name = ["{}/dev/{}".format(data_path,i) for i in dev_name]
print("dev_name",dev_name[10:13])
train_name = train_name1 + train_name
dev_name = dev_name1 + dev_name[:-1]

train_content = train_content1 + train_texts
dev_content = dev_content1 + dev_texts[:-1]

print("load token ")
with open("/workspace/words_audio.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

vocab_size=len(tokenizer.word_index)+1

def greedyDecoder(output,labels,label_lengths,blank_label=1):
    decodes = torch.argmax(output,dim=2)
    #print(decodes.shape)
    targets = tokenizer.sequences_to_texts(labels.detach().cpu().numpy())
    for idx,item in enumerate(label_lengths):
        end_index = 2 * item - 1
        targets[idx] = targets[idx][:end_index]
    bn,seq = decodes.shape
    preds = []
    for i in range(bn):
        decode = []
        for j in decodes[i]:
            if j.item() != blank_label:
                decode.append(j.item())
        preds.append(decode)

    decode_text = tokenizer.sequences_to_texts(preds)
    print("predict: ",decode_text[:2])
    print("targets: ",targets[:2])
    return decode_text, targets
#####
print("init audio_transforms")
audio_dim = 128
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=audio_dim),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)
#shape (channel, n_mels, time):(1,128,379)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def data_collate(data,data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for audio_file,text in data:
        waveform, sample_rate = torchaudio.load(audio_file,normalize=True)
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        label = torch.Tensor(tokenizer.texts_to_sequences([text])[0])
        spectrograms.append(spec)
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


print("init dataloader")

class DataProcesser(Dataset):
    def __init__(self, audio_file, text):
        super(DataProcesser, self).__init__()
        self.first_seg = audio_file
        self.sencond_seg = text

    def __len__(self):
        return len(self.sencond_seg)

    def __getitem__(self, item):
        audio = self.first_seg[item]

        txt = self.sencond_seg[item]

        return (audio,txt)


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val



class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'),save_dir="/model_dir"
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'best_loss':self.best_valid_loss,
                }, '{}/best_model.pth'.format(save_dir))


use_cuda = torch.cuda.is_available()

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    print("training...")
    model.train()
    data_len = len(train_loader.dataset)
    
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))
def test(model, device, test_loader, criterion, epoch, iter_meter):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    # with experiment.test():
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            decoded_preds, decoded_targets = greedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)


    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    return  test_loss , avg_wer
    
def main(learning_rate=5e-4, batch_size=20, epochs=10):

    hparams = {
        "n_cnn_layers": 5,
        "n_rnn_layers": 7,
        "rnn_dim": 512,
        "n_class": vocab_size + 1,
        "n_feats": audio_dim,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    # experiment.log_parameters(hparams)

    
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}    
    
    train_dataset = DataProcesser(train_name,train_content)
    test_dataset = DataProcesser(dev_name,dev_content)
    
    train_loader =DataLoader(dataset=train_dataset  ,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_collate(x, 'train'),
                                **kwargs)
    
    test_loader =DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_collate(x, 'train'),
                                **kwargs)
    
    

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=1,zero_infinity=True).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    
    iter_meter = IterMeter()
    last_model_file = "{}/last_model.pth".format(save_dir)
    best_model_file = "{}/best_model.pth".format(save_dir)
    if os.path.isfile(last_model_file):
        print("load {} model and {} model".format(last_model_file,best_model_file))
        checkpooint = torch.load(last_model_file)
        weights = checkpooint['model_state_dict']
        model.load_state_dict(weights)
        criterion = torch.load(best_model_file)['loss']
        optimizer.load_state_dict(checkpooint['optimizer_state_dict'])
        save_best_model = SaveBestModel(save_dir="/model_dir")
    else:
        print("{} model does't exit ".format(last_model_file))
        save_best_model = SaveBestModel(save_dir="/model_dir")
    
    model=nn.DataParallel(model)
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        test_loss,avg_wer = test(model, device, test_loader, criterion, epoch, iter_meter)
        
        save_best_model(test_loss, epoch, model.module, optimizer, criterion)
        print('-'*50)
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '{}/last_model.pth'.format(save_dir))
    print("finished save finall model")

learning_rate = 5e-4
batch_size = 20
epochs = 200
main(learning_rate, batch_size, epochs)
