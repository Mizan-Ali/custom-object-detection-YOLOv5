import torch
from PIL import Image

# model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')

img = Image.open('dataset/customTests/1.jpg')


result = model(img)
result.render()

result.save('predictions')
result.show()

