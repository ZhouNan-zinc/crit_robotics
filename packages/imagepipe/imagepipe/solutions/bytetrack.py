from typing import Any

import numba
import lap
import numpy as np

def convert_bbox_to_z(bbox):
    """
    Convert the spatial portion of a detection into the Kalman filter measurement.

    Parameters
    ----------
    bbox : array-like
        Detection vector whose first four entries are ``[x1, y1, x2, y2]``.

    Returns
    -------
    np.ndarray
        Column vector ``[x, y, s, r]^T`` where ``(x, y)`` is the box centre,
        ``s`` is the area (scale) and ``r`` is the aspect ratio ``w / h``.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Convert the internal Kalman filter state back to corner coordinates.

    Parameters
    ----------
    x : np.ndarray
        State vector whose first four elements are ``[x, y, s, r]``.
    score : float, optional
        When provided the value is appended as a fifth element.

    Returns
    -------
    np.ndarray
        ``[x1, y1, x2, y2]`` (or ``[x1, y1, x2, y2, score]`` when `score`
        is not ``None``) reshaped to ``(1, -1)``.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))



class KalmanFilter:
    """
    Minimal Kalman filter for the constant-velocity bounding-box model.
    """
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(self):
        """Run the prediction step using the configured dynamics."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Assimilate a measurement vector ``z`` into the filter state."""
        z = np.asarray(z).reshape((self.dim_z, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y

        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x

class TrackingObject:
    """
    Docstring for TrackingObject
    
    :var bbox: Description
    :var x: Description
    :var score: Description
    :vartype score: float
    :var info: Description
    :var detections: Description
    :var trackers: Description
    :var iou_threshold: Description
    :vartype iou_threshold: float
    """

    count = 0

    _F = np.array([
        [1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]
    ], dtype=np.float32)

    _H = np.array([
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0]
    ], dtype=np.float32)

    def __init__(
        self,
        class_id: int | None = None,
        score: float | None = None,
        bbox: np.ndarray | list | None = None,
        mask: np.ndarray | None = None,
        **kwargs
    ):
        self.class_id = class_id
        self.score = score
        self.bbox = np.asarray(bbox) if bbox else None
        self.mask = np.asarray(mask) if mask else None

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.post_init()
    
    def post_init(self):
        self._kf = KalmanFilter(dim_x=7, dim_z=4) 
        self._kf.F = TrackingObject._F
        self._kf.H = TrackingObject._H

        self._kf.R[2:,2:] *= 10.
        self._kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self._kf.P *= 10.
        self._kf.Q[-1,-1] *= 0.01
        self._kf.Q[4:,4:] *= 0.01

        self._kf.x[:4] = convert_bbox_to_z(self.bbox)

        self._time_since_update = 0
        self._history = []
        self._hits = 0
        self._hit_streak = 0
        self._age = 0

        cls = type(self)
        self._id = cls.count
        cls.count += 1

        self.payload : dict[str, Any] = []

    def __getitem__(self, key:str):
        return self.payload.get(key, None)
    
    def __setitem__(self, name:str, value:...):
        if name == "id":
            raise ValueError("Key `id` can not pass to tracker.")
        self.payload[name] = value

    @property
    def id(self) -> int:
        return self._id


    def update(self, obj: "TrackingObject"):
        """
        Update the track state using a fresh detection vector.

        The first four entries of `info` must be ``[x1, y1, x2, y2]``; any
        remaining values (score, class, keypoints, etc.) are stored so that
        downstream consumers receive the most recent detector metadata.
        """

        self.class_id = obj.class_id
        self.score = obj.score
        self.bbox = obj.bbox
        self.payload = obj.payload

        self._time_since_update = 0
        self._history = []
        self._hits += 1
        self._hit_streak += 1
        self._kf.update(convert_bbox_to_z(self.bbox))



    def predict(self):
        """Advance the state vector one frame ahead."""
        if((self._kf.x[6]+self._kf.x[2])<=0):
            self._kf.x[6] *= 0.0
        self._kf.predict()
        self._age += 1

        if(self._time_since_update>0):
            self._hit_streak = 0

        self._time_since_update += 1
        self._history.append(convert_x_to_bbox(self._kf.x))
        return self._history[-1]

    @property
    def get_info(self):
        """
        Return the latest detection info with the Kalman-refined box.

        Returns
        -------
        np.ndarray
            The concatenation of the predicted bbox ``[x1, y1, x2, y2]`` and
            the additional fields supplied by the detector (score, class, ...).
        """
        bbox = convert_x_to_bbox(self._kf.x).reshape(-1)
        return np.concatenate([bbox, self.bbox])


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




def assign(detections: list[TrackingObject], trackers: list[TrackingObject], iou_threshold:float = 0.3):
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
    
    bbox_detections = np.asarray([det.bbox] for det in detections)
    bbox_trackers = np.asarray([trk.bbox] for trk in trackers)
    iou_matrix = iou_batch(
        bbox_detections, bbox_trackers)
    
    class_id_detections = np.asarray([det.class_id] for det in detections)
    class_id_trackers = np.asarray([trk.class_id] for trk in trackers)
    mask = class_id_detections == class_id_trackers # FIXME: mask generated by &

    iou_matrix = iou_matrix & mask

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    matched_detections = detections[matched_indices]
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_indices]

    return matched_detections, unmatched_detections

    # TODO: filter out matched with low IOU
    # matches = []
    # for m in matched_indices:
    #     if(iou_matrix[m[0], m[1]] < iou_threshold):
    #         unmatched_detections.append(m[0])
    #     else:
    #         matches.append(m.reshape(1,2))

    # if(len(matches)==0):
    #     matches = np.empty((0,2),dtype=int)
    # else:
    #     matches = np.concatenate(matches,axis=0)

    # return matches, np.array(unmatched_detections)


class ByteTrack:
    """
    ByteTrack tracker that keeps the full detection info vector per track.
    """
    def __init__(self, max_age=30, min_hits=3, iou_thres=0.3, conf_thres=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.frame_count = 0
        self.trackers: list[TrackingObject] = []

    def update(self, detections: list[TrackingObject]):
        """
        Update the tracker state for one frame.

        """
        self.frame_count += 1

        for idx, trk in reversed(list(enumerate(self.trackers))):
            bbox = trk.predict()
            if np.any(np.isnan(bbox)):
                self.trackers.pop(idx)
            self.trackers[idx].bbox = bbox

        high_score_detections = []
        low_score_detections = []
        
        for det in reversed(detections):
            if det.score >= self.conf_thres:
                high_score_detections.append(detections.pop())
            else:
                low_score_detections.append(detections.pop())

        matched_high, unmatched_high = assign(
            high_score_detections, self.trackers, self.iou_thres)

        for idx, m in enumerate(matched_high):
            self.trackers[idx].update(m)

        if len(unmatched_high) > 0 and len(low_score_detections) > 0:
            matched_low, _ = assign(
                low_score_detections, self.trackers, self.iou_thres)
            for idx, m in enumerate(matched_low):
                self.trackers[idx].update(m)

        self.trackers.extend(unmatched_high)
        return self.trackers
