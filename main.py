#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import sys, getopt
import signal
import time
import numpy as np
from itertools import combinations
from edge_impulse_linux.image import ImageImpulseRunner

runner = None

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()

            print(f"Loaded runner for \"{model_info['project']['owner']} / {model_info['project']['name']}\"")
            labels = model_info['model_parameters']['labels']

            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if not port_ids:
                    raise Exception('Cannot find any webcams')
                if len(args) <= 1 and len(port_ids) > 1:
                    raise Exception('Multiple cameras found. Add the camera port ID as a second argument to this script')
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret, _ = camera.read()
            if ret:
                backend = camera.getBackendName()
                w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera {backend} ({w} x {h}) on port {videoCaptureDeviceId} selected.")
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            def draw_bounding_circles(img: np.ndarray, res, color=(0, 255, 0)) -> None:
                if 'bounding_boxes' not in res['result']:
                    return
                for bb in res['result']['bounding_boxes']:
                    cx = bb['x'] + bb['width'] // 2
                    cy = bb['y'] + bb['height'] // 2
                    radius = int(min(bb['width'], bb['height']) / 2)
                    cv2.circle(img, (cx, cy), radius, color, 2)
                    cv2.putText(img, f"{bb['label']}:{bb['value']:.2f}", (bb['x'], max(0, bb['y'] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            def compute_pairwise_deltas(res):
                if 'bounding_boxes' not in res['result'] or len(res['result']['bounding_boxes']) < 2:
                    return []
                bbs = res['result']['bounding_boxes']
                deltas = []
                for (i, bb_a), (j, bb_b) in combinations(enumerate(bbs), 2):
                    cx_a = bb_a['x'] + bb_a['width'] / 2
                    cy_a = bb_a['y'] + bb_a['height'] / 2
                    cx_b = bb_b['x'] + bb_b['width'] / 2
                    cy_b = bb_b['y'] + bb_b['height'] / 2
                    deltas.append((i, j, cx_b - cx_a, cy_b - cy_a))
                return deltas

            def avg_delta(deltas):
                if not deltas:
                    return None
                mean_dx = sum(d[2] for d in deltas) / len(deltas)
                mean_dy = sum(d[3] for d in deltas) / len(deltas)
                return mean_dx, mean_dy

            def overlay_avg_delta(img: np.ndarray, deltas):
                h = img.shape[0]
                result = avg_delta(deltas)
                if result is None:
                    return
                dx, dy = result
                txt = f"dx={dx:.1f}, dy={dy:.1f}"
                cv2.putText(img, txt, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            def print_classification_and_deltas(res, tag):
                if 'classification' in res['result']:
                    print(f"{tag}: Result ({res['timing']['dsp'] + res['timing']['classification']} ms.) ", end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print(f"{label}: {score:.2f}\t", end='')
                    print('', flush=True)
                elif 'bounding_boxes' in res['result']:
                    bbs = res['result']['bounding_boxes']
                    print(f"{tag}: Found {len(bbs)} bounding boxes ({res['timing']['dsp'] + res['timing']['classification']} ms.)")
                    for bb in bbs:
                        print(f"\t{bb['label']} ({bb['value']:.2f}): x={bb['x']} y={bb['y']} w={bb['width']} h={bb['height']}")
                # Print deltas
                deltas = compute_pairwise_deltas(res)
                if deltas:
                    dx, dy = avg_delta(deltas)
                    print(f"{tag}: avg Δ = (dx={dx:.1f}, dy={dy:.1f})")

            next_frame_time = 0

            for img in runner.get_frames(videoCaptureDeviceId):
                now_ms = now()
                if next_frame_time > now_ms:
                    time.sleep((next_frame_time - now_ms) / 1000)

                feats_l, crop_l = runner.get_features_from_image(img, 'left')
                feats_r, crop_r = runner.get_features_from_image(img, 'right')

                res_l = runner.classify(feats_l)
                res_r = runner.classify(feats_r)

                crop_l_bgr = cv2.cvtColor(crop_l, cv2.COLOR_RGB2BGR)
                crop_r_bgr = cv2.cvtColor(crop_r, cv2.COLOR_RGB2BGR)

                draw_bounding_circles(crop_l_bgr, res_l, (0, 255, 0))      # green
                draw_bounding_circles(crop_r_bgr, res_r, (255, 0, 0))      # blue

                overlay_avg_delta(crop_l_bgr, compute_pairwise_deltas(res_l))
                overlay_avg_delta(crop_r_bgr, compute_pairwise_deltas(res_r))

                combined = np.hstack((crop_l_bgr, crop_r_bgr))
                cv2.imshow('Left (green)  |  Right (blue)', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print_classification_and_deltas(res_l, 'LEFT')
                print_classification_and_deltas(res_r, 'RIGHT')

                next_frame_time = now() + 100  # 10 fps target

        finally:
            if runner:
                runner.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])
