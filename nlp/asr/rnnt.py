import os
import gc
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
from GPUtil import showUtilization as gpu_usage
from torch import Tensor
import GPUtil
from typing import Tuple
import torch
from torchaudio.transforms import RNNTLoss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
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


def load_magic_data(data_dir):
    df = pd.read_csv("{}/TRANS.txt".format(data_dir),sep="\t")
    magic_txt = df['Transcription'].tolist()
    magic_txt = [" ".join([i for i in j]) for j in magic_txt]
    wav_name = df['UtteranceID'].tolist()
    wav_files = []
    for i,speaker in enumerate(df['SpeakerID'].tolist()):
        wav_path = "{}/{}/{}".format(data_dir,speaker,wav_name[i])
        wav_files.append(wav_path)
    return wav_files,magic_txt
print("load first data")
# aishell data load
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
#
data_path = "/workspace/data/audio/speech/waves"

train_name, train_texts = load_audio_file("{}/train.txt".format(data_path))
train_name = ["{}/train/{}".format(data_path,i) for i in train_name]

dev_name, dev_texts = load_audio_file("{}/dev.txt".format(data_path))
dev_name = ["{}/dev/{}".format(data_path,i) for i in dev_name]
# magic data load
magic_data_path = "/workspace/data/audio/magicdata"
test_magic_wav, test_magic_txt = load_magic_data(magic_data_path+"/test")
dev_magic_wav, dev_magic_txt = load_magic_data(magic_data_path+"/dev")
#merge data
train_name = train_name1 + train_name + test_magic_wav
dev_name = dev_name1 + dev_name + dev_magic_wav

train_content = train_content1 + train_texts + test_magic_txt
dev_content = dev_content1 + dev_texts + dev_magic_txt

print("load token ")
with open("/workspace/words_audio.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

vocab_size=len(tokenizer.word_index) 
eos_id = tokenizer.word_index['eos']
sos_id = tokenizer.word_index['sos']



def greedyDecoder(output,labels,label_lengths,blank_label=1):

    #print(decodes.shape)
    label_lengths = label_lengths.detach().cpu().numpy()
    targets = tokenizer.sequences_to_texts(labels.detach().cpu().numpy())
    for idx,item in enumerate(label_lengths):
        end_index = 2 * item - 1
        targets[idx] = targets[idx][:end_index]
    decode_text = tokenizer.sequences_to_texts(output.detach().cpu().numpy())
    
    print("predict: ",decode_text[:2])
    print("targets: ",targets[:2])
    return decode_text, targets




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
        text = ["sos " + text + " eos"]
        label = torch.Tensor(tokenizer.texts_to_sequences(text)[0])
        spectrograms.append(spec)
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = torch.IntTensor(input_lengths)
    label_lengths = torch.IntTensor(label_lengths)
    #spectrograms = spectrograms.squeeze(1).transpose(1,2)
    return spectrograms, labels.type(torch.IntTensor), input_lengths, label_lengths


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



class EncoderRNNT(nn.Module):
    """
    Encoder of RNN-Transducer.
    Args:
        input_dim (int): dimension of input vector
        hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of encoder layers (default: 4)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    Returns:
        (Tensor, Tensor)
        * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,
            hidden_state_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.2,
            bidirectional: bool = True,
    ):
        super(EncoderRNNT, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(hidden_state_dim << 1 if bidirectional else hidden_state_dim, output_dim)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        #print("inputs device:",inputs.device,inputs.shape)
        #print("input length device:",input_lengths)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(),batch_first=True,enforce_sorted=False)
        self.rnn.flatten_parameters()
        outputs, hidden_states = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = self.out_proj(outputs)
        return outputs, input_lengths
    

class DecoderRNNT(nn.Module):
    """
    Decoder of RNN-Transducer
    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of decoder layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout_p (float, optional): dropout probability of decoder
    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    Returns:
        (Tensor, Tensor):
        * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'gru',
            sos_id: int = 1,
            eos_id: int = 2,
            dropout_p: float = 0.2,
    ):
        super(DecoderRNNT, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
        self.out_proj = nn.Linear(hidden_state_dim, output_dim)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
        
    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.
        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            (Tensor, Tensor):
            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        #print(inputs)
        embedded = self.embedding(inputs)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(),batch_first=True, enforce_sorted=False)
            self.rnn.flatten_parameters()
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
            outputs = self.out_proj(outputs)
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.out_proj(outputs)

        return outputs, hidden_states
    
    
class RNNTransducer(nn.Module):
    """
    Args:
        num_classes (int): number of classification
        input_dim (int): dimension of input vector
        num_encoder_layers (int, optional): number of encoder layers (default: 4)
        num_decoder_layers (int, optional): number of decoder layers (default: 1)
        encoder_hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        decoder_hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
        encoder_dropout_p (float, optional): dropout probability of encoder
        decoder_dropout_p (float, optional): dropout probability of decoder
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
    Inputs: inputs, input_lengths, targets, target_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``
    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 1,
            encoder_hidden_state_dim: int = 320,
            decoder_hidden_state_dim: int = 512,
            output_dim: int = 512,
            inner_dim = 768,
            rnn_type: str = "gru",
            bidirectional: bool = True,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            sos_id: int = 1,
            eos_id: int = 2,
            num_gpus: int = 1
    ):
        super(RNNTransducer, self).__init__()
        self.num_classes = num_classes
        self.num_gpus = num_gpus
        self.encoder = EncoderRNNT(
            input_dim=input_dim,
            hidden_state_dim=encoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_encoder_layers,
            rnn_type=rnn_type,
            dropout_p=encoder_dropout_p,
            bidirectional=bidirectional,
        )
        self.decoder = DecoderRNNT(
            num_classes=num_classes,
            hidden_state_dim=decoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
            sos_id=sos_id,
            eos_id=eos_id,
            dropout_p=decoder_dropout_p,
        )
        self.tanh =  nn.Tanh()
        self.forward_laryer = nn.Linear(decoder_hidden_state_dim * 2,inner_dim)
        self.fc = nn.Linear(inner_dim, num_classes, bias=False)

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of model """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)
        
    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        #print("concat",outputs.shape)
        #print("device",outputs.device)
        outputs = self.forward_laryer(outputs)
        outputs = self.tanh(outputs)
        outputs = self.fc(outputs)
        #shape (batch,max input_len, max target_len + 1,classes)or (b,t,u+1,classes)
        return outputs 

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
            max_input_len: int = None,
            max_target_len: int = None
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        #input_lengths,target_lengths = input_lengths.to(inputs.device),target_lengths.to(targets.device)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(concat_targets, target_lengths.add(1))
        if self.num_gpus > 1:
            input_pad_len = max_input_len - encoder_outputs.size(1)
            encoder_outputs = F.pad(encoder_outputs.transpose(1,2),pad=(0,input_pad_len),value=0)
            encoder_outputs = encoder_outputs.transpose(1,2)

            tar_pad_len = max_target_len - decoder_outputs.size(1)
            decoder_outputs = F.pad(decoder_outputs.transpose(1,2),pad=(0,tar_pad_len),value=0)
            decoder_outputs = decoder_outputs.transpose(1,2)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)
        
        return torch.LongTensor(pred_tokens)  # shape ()

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions. 
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs
    

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
    train_loss = 0.0
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        #print("input shape",spectrograms.shape)
        bn,seq_size,dim = spectrograms.shape 
        input_lengths,label_lengths = input_lengths.to(device),label_lengths.to(device)
        max_input_len = input_lengths.max().item()
        max_target_len = label_lengths.max().item() + 1
        optimizer.zero_grad()
        output = model(spectrograms,input_lengths,labels,label_lengths,max_input_len,max_target_len)  # (batch, max_input_length,max_target_length + 1, n_class)
        #GPUtil.showUtilization()
        #print("after model train")
        #gpu_usage()
        output = F.log_softmax(output, dim=-1)
        loss = criterion(output, labels, input_lengths, label_lengths)
        #print("cal loss gpu usage")
        #gpu_usage()
        loss.backward()
        #print("after loss backward")
        #gpu_usage()
        optimizer.step()
        scheduler.step()
        iter_meter.step()
        train_loss += loss.item() / data_len
        
            #print("spectrogram shape:",(bn,seq_size,dim))
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms),
                data_len,100. * batch_idx / len(train_loader), loss.item()))
        del output,loss,spectrograms,labels,input_lengths,label_lengths
        #del output,loss,spectrograms,labels,input_lengths,label_lengths
        gc.collect()
        torch.cuda.empty_cache()
    print("Train Epoch:{}\tAvg_loss:{:.6f}".format(epoch,train_loss))
        #print("after empty cache")
        #gpu_usage()
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
            input_lengths,label_lengths = input_lengths.to(device),label_lengths.to(device)
            max_inputs_length = input_lengths.max().item()
            max_target_length = label_lengths.max().item() + 1
            output = model(spectrograms,input_lengths,labels,label_lengths,max_inputs_length,max_target_length)  # (batch, max_input_length,max_target_length + 1, n_class)
            output = F.log_softmax(output, dim=-1)
            
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            decoded_preds = model.recognize(spectrograms,input_lengths)
            decoded_preds, decoded_targets = greedyDecode(decoded_preds,labels,label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)


    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    return  test_loss , avg_wer




def main(learning_rate=5e-4, batch_size=20, epochs=10):

    hparams = {
        "encoder_layers": 2,
        "decoder_layers": 2,
        "input_dim": audio_dim,
        "n_class": vocab_size + 1,
        "n_feats": audio_dim,
        "stride":2,
        "output_dim":512,
        "inner_dim":768,
        "encoder_hidden_state_dim":256,
        "decoder_hidden_state_dim":512,
        "dropout": 0.1,
        "num_gpus":4,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    # experiment.log_parameters(hparams)

    
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}    
    
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
                                collate_fn=lambda x: data_collate(x, 'valid'),
                                **kwargs)
    
    

    model = RNNTransducer(num_classes=hparams['n_class'],
            input_dim=hparams['input_dim'],
            sos_id = sos_id,
            eos_id = eos_id,
            num_decoder_layers = hparams['decoder_layers'],
            num_gpus=4)
    print(model)
    print('Num Model Parameters', model.count_parameters())
    
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = RNNTLoss(blank=0)
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
                'model_state_dict': model.module.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '{}/last_model.pth'.format(save_dir))

learning_rate = 5e-4
batch_size = 4
epochs = 50
main(learning_rate, batch_size, epochs)
