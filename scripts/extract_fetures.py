from torchreid.utils import FeatureExtractor
import torch

import os
import sys
import numpy as np
import cv2

from PIL import Image

extractor = FeatureExtractor(
    model_name='resnet50_fc512',
    model_path='/home/dmitriy.khvan/deep-person-reid/resnet50_fc512_market_xent.pth.tar',
    device='cuda'
)

def test_sanity():
    features = extractor('/home/dmitriy.khvan/deep-person-reid/tmp/debug/dbg_orig_0.jpg')
    print(features)

def extract_image_patch(image, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    patch = image[y:y+h, x:x+w]
    return patch

def resnet50_feed(image, boxes, model):
    features_out = []

    for num, box in enumerate(boxes):
        patch = extract_image_patch(image, box)
        # cv2.imwrite('./tmp/debug/dbg_orig_%d.jpg' % (num), patch)
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if patch_rgb is None:
            print("WARNING: Failed to extract image patch: %s." % str(box))
            patch_rgb = np.random.uniform(0., 255., image_shape).astype(np.uint8)

        patch_np = np.array(patch_rgb)
        features = extractor(patch_np)
        # print (type(features), features.size())

        features = torch.flatten(features)
        features_np = features.cpu().detach().numpy()

        features_out.append(features_np)

    features_out = np.asarray(features_out)
    # print(type(features_out), features_out.shape)
    
    return features_out

def generate_detections(ckpt_path, mot_dir, output_dir, detection_dir=None):

    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(mot_dir, "img1")

        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(mot_dir, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
        
            bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)

            features = resnet50_feed(bgr_image, rows[:, 2:6].copy(), None)
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)
        break


if __name__=="__main__": 
    if sys.argv[1] == 'extract':
        generate_detections(sys.argv[2], '/home/dmitriy.khvan/dsort-gcp/bepro-data/data/', '/home/dmitriy.khvan/dsort-gcp/bepro-data/out')
