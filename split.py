from pydub import AudioSegment
import math
import os
progpath=os.path.dirname(os.path.realpath(__file__))
foldname = input('enter uid')
class SplitWavAudioMubin():
    def __init__(self):
        
        self.folder = os.path.join(progpath, '16000_pcm_speeches', 'audio', foldname)
        self.filename = 'output.wav'
        self.filepath = self.folder + '\\' + self.filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_sec, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_sec * 1000 * 60
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration()/60)
        i=0
        while i < total_mins:
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')
            i+=min_per_split
a = SplitWavAudioMubin()
a.multiple_split(1/60)
os.remove(os.path.join(progpath, '16000_pcm_speeches', 'audio', foldname, 'output.wav'))