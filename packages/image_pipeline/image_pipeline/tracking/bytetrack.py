import lap
import numba
import numpy as np

from .kalman_tracker import KalmanBoxTracker

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem used to match detections with trackers.
    """
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #


def iou_batch(bb_test, bb_gt):
    """
    Compute pairwise IOU for arrays of `[x1, y1, x2, y2]` boxes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  


@numba.njit(fastmath=True)
def iou_batch_jit(bb_test, bb_gt):
    """
    Compute IOU with a numba-optimised double loop for large batch counts.
    """
    num_test = bb_test.shape[0]
    num_gt = bb_gt.shape[0]
    result = np.zeros((num_test, num_gt), dtype=np.float64)

    for i in range(num_test):
        x1 = bb_test[i, 0]
        y1 = bb_test[i, 1]
        x2 = bb_test[i, 2]
        y2 = bb_test[i, 3]
        area_test = (x2 - x1) * (y2 - y1)

        for j in range(num_gt):
            gx1 = bb_gt[j, 0]
            gy1 = bb_gt[j, 1]
            gx2 = bb_gt[j, 2]
            gy2 = bb_gt[j, 3]
            area_gt = (gx2 - gx1) * (gy2 - gy1)

            xx1 = x1 if x1 > gx1 else gx1
            yy1 = y1 if y1 > gy1 else gy1
            xx2 = x2 if x2 < gx2 else gx2
            yy2 = y2 if y2 < gy2 else gy2

            w = xx2 - xx1
            if w <= 0.0:
                continue
            h = yy2 - yy1
            if h <= 0.0:
                continue

            inter = w * h
            union = area_test + area_gt - inter
            if union <= 0.0:
                result[i, j] = 0.0
            else:
                result[i, j] = inter / union

    return result




def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assign detections to tracker predictions using an IOU threshold.

    Parameters
    ----------
    detections : np.ndarray
        Array of boxes ``[N, 4]`` in ``[x1, y1, x2, y2]`` format.
    trackers : np.ndarray
        Array of tracker predictions ``[M, 4]``.
    iou_threshold : float
        Minimum IOU for an association to be accepted.

    Returns
    -------
    tuple
        ``(matches, unmatched_detections)`` with ``matches`` shaped ``[K, 2]``.
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    matched_detections = set(matched_indices[:, 0])
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_detections]

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections)


class ByteTrack(object):
    """
    ByteTrack tracker that keeps the full detection info vector per track.
    """
    def __init__(self, max_age=30, min_hits=3, iou_thres=0.3, conf_thres=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.frame_count = 0
        self.trackers:list[KalmanBoxTracker] = []

    def update(self, dets=np.empty((0, 6))):
        """
        Update the tracker state for one frame.

        Parameters
        ----------
        dets : np.ndarray, optional
            Detection tensor ``[N, K]`` whose rows contain the bounding box,
            confidence, class id, and any additional per-detection attributes.
            All columns beyond the first four are preserved and propagated in
            the track output so downstream consumers receive the detector info.

        Returns
        -------
        tuple[list[int], list[np.ndarray]]
            ``ids`` is the list of active track identifiers in the same order
            as the rows of ``tracks``. ``tracks`` contains the Kalman-refined
            bounding boxes concatenated with the detector payload (score, class,
            keypoints, ...). An empty tracks array is returned when no track is
            active, but the ``ids`` list is still provided (usually empty).

        Notes
        -----
        The method predicts every tracker forward, associates high-confidence
        detections first, uses low-confidence detections to recover unmatched
        tracks, spawns new trackers from unmatched confident detections, and
        finally removes trackers that have aged past ``max_age``. Track info
        arrays always mirror the detector schema so consumers can trust column
        ordering.
        """
        self.frame_count += 1

        dets = np.asarray(dets, dtype=float)
        if dets.ndim == 1:
            dets = dets[None, :]

        N = len(self.trackers)
        trks = np.empty((N, 5), dtype=np.float32)
        valid = np.ones(N, dtype=bool)

        for i, tracker in enumerate(self.trackers):
            pos = tracker.predict()[0]

            if np.any(np.isnan(pos)):
                valid[i] = False
                continue

            trks[i, :4] = pos
            trks[i, 4] = 0.

        trks = trks[valid]
        for i in range(N-1, -1, -1):
            if not valid[i]:
                self.trackers.pop(i)

        scores = dets[:, 4]
        dets_high = dets[scores >= self.conf_thres, :]
        dets_low = dets[scores < self.conf_thres, :]
        boxes_high = dets_high[:, :4]
        boxes_low = dets_low[:, :4]
        tracker_predictions = trks[:, :4]

        matched_high, unmatched_high = associate_detections_to_trackers(
            boxes_high, tracker_predictions, self.iou_thres)

        tracker_matched = np.zeros(len(self.trackers), dtype=bool)
        for det_idx, trk_idx in matched_high:
            tracker_matched[trk_idx] = True
            self.trackers[trk_idx].update(dets_high[det_idx])

        unmatched_tracker_idx = np.where(~tracker_matched)[0]
        if len(unmatched_tracker_idx) > 0 and len(boxes_low) > 0:
            secondary_predictions = tracker_predictions[unmatched_tracker_idx]
            matched_low, _ = associate_detections_to_trackers(
                boxes_low, secondary_predictions, self.iou_thres)
            for det_idx, sub_idx in matched_low:
                trk_idx = unmatched_tracker_idx[sub_idx]
                tracker_matched[trk_idx] = True
                self.trackers[trk_idx].update(dets_low[det_idx])

        for idx in unmatched_high:
            if len(boxes_high) == 0:
                break
            trk = KalmanBoxTracker(dets_high[idx])
            self.trackers.append(trk)

        ids = []
        ret = []
        idx = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ids.append(trk.get_id())
                ret.append(trk.get_info().reshape(1, -1))
            idx -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(idx)

        return ids, ret
