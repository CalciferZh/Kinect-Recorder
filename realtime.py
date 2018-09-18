from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
import numpy as np
import pygame
import time
import queue
import threading
import cv2


kinect = PyKinectRuntime.PyKinectRuntime(
  PyKinectV2.FrameSourceTypes_Depth |
  PyKinectV2.FrameSourceTypes_Color
)
depth_height = kinect.depth_frame_desc.Height
depth_width = kinect.depth_frame_desc.Width
color_height = kinect.color_frame_desc.Height
color_width = kinect.color_frame_desc.Width
fps = 30
recording = False
min_fps = 30
color = np.zeros([color_height, color_width, 3])
depth = np.zeros([depth_height, depth_width])
done = False
save_prefix = input('Please enter save_prefix: ')
color_file = None
color_video = cv2.VideoWriter(
  save_prefix + '_color.avi',
  cv2.VideoWriter_fourcc(*'DIVX'),
  fps,
  (color_width, color_height)
)
depth_frames = []
num_frames = 0
start_time = None
end_time = None

pygame.init()
screen = pygame.display.set_mode(
  (640, 480),
  pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
  24
)
clock = pygame.time.Clock()
pygame.display.set_caption('Kinect Recorder')

q = queue.Queue()
t = None

def save_frame():
  while True:
    print(q.qsize())
    frame = q.get()
    if frame is None:
      break
    # frame.tofile(color_file)
    frame = np.reshape(
      frame,
      [color_height, color_width, -1]
    )
    frame = frame[..., :3]
    color_video.write(frame)
    q.task_done()

while not done:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      done = True
    elif event.type == pygame.KEYDOWN:
      if recording:
        pass
      else:
        recording = True
        pygame.display.set_caption('Kinect Recorder - Recording')
        color_file = open(save_prefix + '_color.data', 'w')
        # color_file = op
        start_time = time.time()
        t = threading.Thread(target=save_frame)
        t.start()

  color = kinect.get_last_color_frame()
  depth = kinect.get_last_depth_frame()
  if recording:
    q.put(color)
    depth_frames.append(depth)
    fps = clock.get_fps()
    num_frames += 1
    if fps != 0 and fps < min_fps:
      min_fps = fps
  clock.tick(fps)
end_time = time.time()
pygame.quit()

q.put(None)

t.join()

# color_file.close()
color_video.release()

print("Minimal FPS: %f" % min_fps)
print("Average FPS: %f" % (num_frames / (end_time - start_time)))

