import whisper
from whisper.utils import get_writer
writer_txt = get_writer("vtt",'result')
model = whisper.load_model("/root/moses/whisper/pretrain/large-v2.pt")
result = model.transcribe("data/test.mp3",task='transcribe')

options={"max_line_width":None,"max_line_count":None,"highlight_words":""}
with open("sub.vtt","w") as f:
    writer_txt.write_result(result,f,options)
