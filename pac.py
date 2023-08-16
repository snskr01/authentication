import os
progpath=os.path.dirname(os.path.realpath(__file__))
foldname = input('enter uid')
filename = input('enter file name with extension')
source_file = os.path.join(progpath, '16000_pcm_speeches', 'audio', foldname, filename)
output_file = os.path.join(progpath, '16000_pcm_speeches', 'audio', foldname, 'output.wav')
cmd_str=f"ffmpeg -i {source_file} -ac 1 -ar 16000 {output_file}"
print(cmd_str)
os.system(cmd_str)