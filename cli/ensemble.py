import os
import cv2
import glob
import json
import copy
import pandas as pd
import numpy as np
import itertools
import datetime


def computeIOU(box1, box2):
    x11, y11, x12, y12 = box1[1:]
    x21, y21, x22, y22 = box2[1:]

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


# In[ ]:


def draw_box(img, coords, color=[255, 0, 0]):
    """
    Function to draw bounding boxes on image.
    """
    img = img.copy()
    #     coords = coords.reshape(-1, 4)
    for coord in coords:
        ul_pt = (int(coord[0]), int(coord[1]))  # Upper left point
        br_pt = (int(coord[2]), int(coord[3]))  # Bottom right point      
        img = cv2.rectangle(img.copy(), ul_pt, br_pt, color, int(max(img.shape[:2]) / 200))
    return img


# In[ ]:


def convert_coords(coords):
    new_coords = []
    for coord in coords:
        _coord = copy.copy(coord)
        _coord[3] = coord[1] + coord[3]
        _coord[4] = coord[2] + coord[4]
        _coord = list(map(float, _coord))
        _coord[1:] = list(map(int, _coord[1:]))
        new_coords.append(_coord)
    return new_coords


# In[ ]:


def find_coord(coord, boxes):
    if all(isinstance(item, list) for item in boxes):
        return any(find_coord(coord, s) for s in boxes)
    return coord == boxes


# In[ ]:


def get_overlapping_pairs(coords, threshold):
    overlapping_boxes = []
    for coord1, coord2 in itertools.combinations(coords, 2):
        iou = computeIOU(coord1, coord2)
        if iou > threshold:
            overlapping_boxes.append([coord1, coord2])
    for coord in coords:
        if not find_coord(coord, overlapping_boxes):
            overlapping_boxes.append([coord])
    return overlapping_boxes


# In[ ]:


def iter_group(iterator, count):
    itr = iter(iterator)
    while True:
        yield list([itr.__next__() for i in range(count)])


# In[ ]:


def read_submission_file(file_list):
    df_dict = []
    for file in file_list:
        with open(file) as fh:
            lines = fh.readlines()
        for line in lines[1:]:
            line = line.strip()
            image_id, boxes = line.split(",")
            if len(boxes):
                for box in iter_group(boxes.split(), 5):
                    box = list(map(float, box))
                    box[1:] = list(map(int, box[1:]))
                    df_dict.append({'patientId': image_id, "PredictionString": box})
            else:
                df_dict.append({'patientId': image_id, "PredictionString": []})

    return pd.DataFrame.from_dict(df_dict)


# In[ ]:


def get_all_predictionboxes(patient_id):
    _pred_df = pred_df.loc[pred_df["patientId"] == patient_id]
    boxes = [x for x in list(_pred_df["PredictionString"]) if x != []]
    return boxes


# In[ ]:


def get_index_if_found(pair, library):
    index = -1
    for i, group in enumerate(library):
        for p in pair:
            if p in group:
                return i
    return index


# In[ ]:


def merge_overlapping_boxes(over_pairs):
    _merged_boxes = []
    for pair in over_pairs:
        index = get_index_if_found(pair, _merged_boxes)
        if index == -1:
            _merged_boxes.append(pair)
        else:
            _merged_boxes[index] = _merged_boxes[index] + pair
    merged_boxes = []
    for index in range(len(_merged_boxes)):
        merged_boxes.append([])
        for p in _merged_boxes[index]:
            if p not in merged_boxes[index]:
                merged_boxes[index].append(p)
    return merged_boxes


# In[ ]:


def get_ensemble_boxes(file_list, boxes, alg="max", stdmult=0, overlap_threshold=0.0,
                       scoring_ensemble_method="boxes_mean", mean_score_threshold=0.0):
    overlapping_pairs = get_overlapping_pairs(boxes, overlap_threshold)
    #     print(overlapping_boxes)
    merged_boxes = merge_overlapping_boxes(overlapping_pairs)
    #     print(merged_boxes)
    return_boxes = []
    return_scores = []
    for b in merged_boxes:
        barray = np.array(b)
        if len(barray):
            scores = barray[:, 0]
            boxes_array = barray[:, 1:]
            if scoring_ensemble_method == "number_mean":
                mean_score = np.mean(scores)
            elif scoring_ensemble_method == "files_mean":
                mean_score = np.sum(scores) / len(file_list)
            if mean_score >= mean_score_threshold:
                return_scores.append(mean_score)
                if alg == 'max':
                    return_boxes.append([np.min(boxes_array[:, 0], axis=0),
                                         np.min(boxes_array[:, 1], axis=0),
                                         np.max(boxes_array[:, 2], axis=0),
                                         np.max(boxes_array[:, 3], axis=0), ])
                elif alg == 'mean':

                    return_boxes.append(
                        [np.mean(boxes_array[:, 0], axis=0) - stdmult * np.std(boxes_array[:, 0], axis=0),
                         np.mean(boxes_array[:, 1], axis=0) - stdmult * np.std(boxes_array[:, 1], axis=0),
                         np.mean(boxes_array[:, 2], axis=0) + stdmult * np.std(boxes_array[:, 2], axis=0),
                         np.mean(boxes_array[:, 3], axis=0) + stdmult * np.std(boxes_array[:, 3], axis=0), ])
                elif alg == 'median':
                    return_boxes.append(
                        [np.median(boxes_array[:, 0], axis=0) - stdmult * np.std(boxes_array[:, 0], axis=0),
                         np.median(boxes_array[:, 1], axis=0) - stdmult * np.std(boxes_array[:, 1], axis=0),
                         np.median(boxes_array[:, 2], axis=0) + stdmult * np.std(boxes_array[:, 2], axis=0),
                         np.median(boxes_array[:, 3], axis=0) + stdmult * np.std(boxes_array[:, 3], axis=0), ])
    return return_boxes, return_scores


if __name__ == '__main__':
    ensemble_config = json.loads(open("ENSEMBLE_SETTINGS.json").read())
    scoring_ensemble_method = ensemble_config["scoring_ensemble_method"]
    mean_score_threshold = ensemble_config["mean_score_threshold"]
    ensemble_method = ensemble_config["ensemble_method"]
    overlap_threshold = ensemble_config["overlap_threshold"]
    std_multiplier = ensemble_config["std_multiplier"]
    base_paths = [ensemble_config["SUBMISSION_DIR"]]
    output_directory = ensemble_config["ENSEMBLE_DIR"]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # path to submission files generated by various models
    suffix = ""
    for p in base_paths:
        suffix += "_" + os.path.basename(p)
    output_filename = os.path.join(output_directory,"ens_sub_over-th-%0.2f_mean-th-%0.2f_"
                                                    "box-en-%s_score-en-%s_std-%.2f%s_%s.txt" % (
        overlap_threshold, mean_score_threshold, ensemble_method, scoring_ensemble_method, std_multiplier,
        suffix, datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")))

    file_list = []
    for path in base_paths:
        file_list += glob.glob(os.path.join(path, "*txt"))

    pred_df = read_submission_file(file_list)
    outh = open(output_filename, 'w')
    outh.write("patientId,PredictionString\n")
    for u in pred_df["patientId"].unique():
        prediction_boxes = convert_coords(get_all_predictionboxes(u))
        ensemble_boxes, ensemble_scores = get_ensemble_boxes(file_list, prediction_boxes, alg=ensemble_method,
                                                             stdmult=std_multiplier,
                                                             overlap_threshold=overlap_threshold,
                                                             scoring_ensemble_method=scoring_ensemble_method,
                                                             mean_score_threshold=mean_score_threshold)
        outh.write("%s," % u)
        for b, s in zip(ensemble_boxes, ensemble_scores):
            b[2] = b[2] - b[0]
            b[3] = b[3] - b[1]
            outh.write(" %f %d %d %d %d" % tuple([s] + b))
        outh.write("\n")
    outh.close()
