from detect import detect
from PIL import Image
import os
import glob
from tqdm import tqdm

# Get frames from video
video_folder = './videos'
video_name = 'Homeward Bound II  Lost in San Francisco Airport Escape.mp4'
video_path = os.path.join(video_folder, video_name)
temp_folder = os.path.join(video_folder, 'temp')
os.system("mkdir '" + temp_folder + "'")
frames_path = os.path.join(temp_folder, 'frames%d.jpg')
intervals = [(0, 150)]
fps = 25

intervals = ['between(t\,{}\,{})+'.format(interval[0], interval[1]) for interval in intervals][0][:-1]

command = "ffmpeg -i '" + video_path + "' -vf fps=" + str(
    int(fps)) + ",select='" + intervals + "' -vsync 0 '" + frames_path + "'"
os.system(command)

for img_path in tqdm(glob.glob(os.path.join(temp_folder, 'frames*.jpg'))):
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    annotated_image = detect(original_image, min_score=None, max_overlap=0.45, top_k=200)
    annotated_image.save(img_path)

output_video_path = os.path.join(video_folder, "annotated_" + video_name)
os.system("ffmpeg -r " + str(int(fps)) + " -i '" + frames_path + "' -c:v libx264 '" + output_video_path + "'")

os.system("rm -r '" + temp_folder + "'")
