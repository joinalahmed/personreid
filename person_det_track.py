import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment
import helpers
import detector
import tracker
import cv2
import pickle

frame_count = 0  # frame counter
max_age = 15  # no.of consecutive unmatched detection before  track is deleted
min_hits = 1  # no. of consecutive matches needed to establish a track
tracker_list = []  # list for trackers
track_id = 0
track_id_list = []
final_dict = {}
debug = False


def assign_detections_to_trackers(trackers, detections, iou_thresh=0.3):
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, detection in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.box_iou2(trk, detection)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if t not in matched_idx[:, 0]:
            unmatched_trackers.append(t)

    for d, detection in enumerate(detections):
        if d not in matched_idx[:, 1]:
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object

    for m in matched_idx:
        if IOU_mat[m[0], m[1]] < iou_thresh:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(img_in):
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global track_id

    frame_count += 1

    z_box = det.get_localization(img_in)  # measurement
    if debug:
        print('Frame:', frame_count)

    x_box = []

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thresh=0.3)
    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)

    # Deal with matched detections     
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1

    # Deal with unmatched detections      
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            track_id += 1
            tmp_trk.id = track_id  # assign an ID for the tracker
            print(tmp_trk.id)
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks       
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # The list of tracks to be annotated  
    good_tracker_list = []
    for trk in tracker_list:
        if (trk.hits >= min_hits) and (trk.no_losses <= max_age):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            if debug:
                print('updated box: ', x_cv2)
            img_in,img_crop = helpers.draw_box_label(trk.id, img_in, x_cv2, frame_count)
            if img_crop is not None:
                img_trk=[img_in,img_crop,x_cv2]
                if trk.id in final_dict.keys():

                    l = final_dict[trk.id]

                    l.append(img_trk)

                    final_dict.update({trk.id: l})

                else:

                    l = []

                    l.append(img_trk)

                    final_dict.update({trk.id: l})

    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]
    fileObject = open("final_dict_pickle", 'wb')
    pickle.dump(final_dict, fileObject)
    fileObject.close()
    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    cv2.imshow("frame", img_in)
    return img_in


if __name__ == "__main__":

    det = detector.PersonDetector()
    cap = cv2.VideoCapture('/home/leadics-18-2/Desktop/Unique_People_Identification/3dec12.mp4')
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    video_object = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output_exp.avi', video_object, 8.0, (int(cap.get(3)), int(cap.get(4))))
    while frame_num:
        ret, img = cap.read()
        frame_num = frame_num - 1
        if not ret:
            continue
        np.asarray(img)
        new_img = pipeline(img)
        out.write(new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
