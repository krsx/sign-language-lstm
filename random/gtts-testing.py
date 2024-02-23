# from gtts import gTTS
# import os

# # Teks yang ingin diubah menjadi ucapan
# text = "Halo, ini adalah contoh teks yang akan diubah menjadi ucapan dalam Bahasa Indonesia."

# # Membuat objek gTTS
# tts = gTTS(text, lang='id')

# # Menyimpan file audio
# tts.save("output.mp3")

# # Memutar file audio
# os.system("start output.mp3")

# from gtts import gTTS
# from tempfile import TemporaryFile
# tts = gTTS(text='Halo semuanya nama saya Krisna Erlangga', lang='id')
# f = TemporaryFile()
# tts.write_to_fp(f)
# # Play f
# f.close()

from gtts import gTTS
from time import sleep
import os
import pyglet

sentence = ["Krisna", "Erlangga"]

tts = gTTS(text=' '.join(sentence), lang='id')
filename = '/tmp/temp.mp3'
tts.save(filename)

music = pyglet.media.load(filename, streaming=False)
music.play()

sleep(music.duration)  # prevent from killing
os.remove(filename)  # remove temperory file
