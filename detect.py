from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)



    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return original_image

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        draw.rectangle(xy=det_boxes[i].tolist(), outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=(det_boxes[i] + 1.).tolist(), outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.text(xy=(det_boxes[i] + 10.).tolist()[:2], text=det_labels[i], fill=label_color_map[det_labels[i]])
    del draw

    return annotated_image


if __name__ == '__main__':
    img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000229.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.21, max_overlap=0.45, top_k=200).show()
