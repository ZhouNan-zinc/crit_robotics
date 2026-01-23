from typing import Any

import cv2
import numba
import logging
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

    arr = np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape(-1)
    if score is None:
        return arr.astype(np.float32)
    return np.concat([arr, np.array([score], dtype=np.float32)])


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
        z = np.asarray(z).reshape((self.dim_z, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # --- CHANGED: use solve() instead of pinv() for speed and stability ---
        # We want: K = P H^T S^{-1}
        # Solve S * X = (H P)^T  => X = S^{-1} (H P)^T
        PHt = self.P @ self.H.T                           # (dim_x, dim_z)
        Sinv_PHtT = np.linalg.solve(S, PHt.T).T            # (dim_x, dim_z)
        K = Sinv_PHtT

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
    :var iou_thres: Description
    :vartype iou_thres: float
    """

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
        self.bbox = np.asarray(bbox) if bbox is not None else None
        self.mask = np.asarray(mask) if mask is not None else None
        
        self._payload = set()

        for k, v in kwargs.items():
            self._payload.add(k)
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
        self._hit_streak = 0
        self._age = 0
        self._id = None

    @property
    def id(self) -> int|None:
        return self._id
    
    @id.setter
    def id(self, value: int | None):
        self._id = int(value) if value is not None else value

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
        self.mask = obj.mask
        
        for k in obj._payload:
            setattr(self, k, getattr(obj, k))

        self._time_since_update = 0
        self._history = []
        self._hit_streak += 1
        self._kf.update(convert_bbox_to_z(self.bbox))


    def predict(self):
        """Advance the state vector one frame ahead."""
        if((self._kf.x[6]+self._kf.x[2])<=0):
            self._kf.x[6] *= 0.0
        self._kf.predict()
        self._age += 1

        if(self._time_since_update > 0):
            self._hit_streak = 0

        self._time_since_update += 1
        self._history.append(convert_x_to_bbox(self._kf.x))
        return self._history[-1]


def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem used to match detections with trackers.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float32)
    _, x, _ = lap.lapjv(cost_matrix, extend_cost=True)

    matched = []
    for row, col in enumerate(x):
        if col >= 0:
            matched.append([row, col])   # row=det_idx, col=trk_idx
    return np.asarray(matched, dtype=int)


@numba.njit(fastmath=True, cache=True)
def generalized_iou(a, b):
    n = a.shape[0]
    m = b.shape[0]
    result = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        ax1 = a[i, 0]
        ay1 = a[i, 1]
        ax2 = a[i, 2]
        ay2 = a[i, 3]

        aw = ax2 - ax1
        ah = ay2 - ay1
        if aw <= 0.0 or ah <= 0.0:
            a_area = 0.0
        else:
            a_area = aw * ah

        for j in range(m):
            bx1 = b[j, 0]
            by1 = b[j, 1]
            bx2 = b[j, 2]
            by2 = b[j, 3]

            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0.0 or bh <= 0.0:
                b_area = 0.0
            else:
                b_area = bw * bh

            inter_w = min(ax2, bx2) - max(ax1, bx1)
            inter_h = min(ay2, by2) - max(ay1, by1)
            if inter_w <= 0.0 or inter_h <= 0.0:
                inter = 0.0
            else:
                inter = inter_w * inter_h

            union = a_area + b_area - inter
            if union <= 0.0:
                iou = 0.0
            else:
                iou = inter / union

            cx1 = ax1 if ax1 < bx1 else bx1
            cy1 = ay1 if ay1 < by1 else by1
            cx2 = ax2 if ax2 > bx2 else bx2
            cy2 = ay2 if ay2 > by2 else by2

            cw = cx2 - cx1
            ch = cy2 - cy1
            if cw <= 0.0 or ch <= 0.0:
                result[i, j] = iou
            else:
                c_area = cw * ch
                result[i, j] = iou - (c_area - union) / c_area

    return result


def assign(detections: list[TrackingObject],
           trackers: list[TrackingObject],
           iou_thres: float = 0.3):

    D = len(detections)
    T = len(trackers)

    if D == 0 or T == 0:
        matched = np.empty((0, 2), dtype=int)
        unmatched_detections = np.arange(D, dtype=int)
        unmatched_trackers = np.arange(T, dtype=int)
        return matched, unmatched_detections, unmatched_trackers

    bbox_detections = np.asarray([det.bbox for det in detections], dtype=np.float32).reshape(-1, 4)
    bbox_trackers   = np.asarray([trk.bbox for trk in trackers], dtype=np.float32).reshape(-1, 4)

    class_id_detections = np.asarray([det.class_id for det in detections], dtype=np.int32)
    class_id_trackers   = np.asarray([trk.class_id for trk in trackers], dtype=np.int32)

    iou_matrix = np.zeros((D, T), dtype=np.float32)

    classes = np.intersect1d(np.unique(class_id_detections), np.unique(class_id_trackers))
    for c in classes:
        det_idx = np.where(class_id_detections == c)[0]
        trk_idx = np.where(class_id_trackers == c)[0]
        if det_idx.size == 0 or trk_idx.size == 0:
            continue

        # compute IOU only for this class block
        block_iou = generalized_iou(bbox_detections[det_idx], bbox_trackers[trk_idx])
        iou_matrix[np.ix_(det_idx, trk_idx)] = block_iou

    # Assignment
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_thres).astype(np.int32)

        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1).astype(int)
        else:
            matched_indices = linear_assignment(-iou_matrix)
            matched_indices = np.asarray(matched_indices, dtype=int).reshape(-1, 2)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    if len(matched_indices) > 0:
        pair_ious = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]
        keep = pair_ious >= iou_thres
        matched = matched_indices[keep]
    else:
        matched = np.empty((0, 2), dtype=int)

    matched_det_ids = matched[:, 0] if len(matched) > 0 else np.array([], dtype=int)
    matched_trk_ids = matched[:, 1] if len(matched) > 0 else np.array([], dtype=int)

    det_mask = np.zeros(D, dtype=bool)
    trk_mask = np.zeros(T, dtype=bool)
    det_mask[matched_det_ids] = True
    trk_mask[matched_trk_ids] = True

    unmatched_detections = np.where(~det_mask)[0].astype(int)
    unmatched_trackers   = np.where(~trk_mask)[0].astype(int)

    return matched, unmatched_detections, unmatched_trackers


class ByteTrack:
    """
    ByteTrack tracker that keeps the full detection info vector per track.
    """

    count = 0

    def __init__(
        self,
        max_age:int|None=None,
        min_hits:int|None=None,
        iou_thres:float|None = None,
        conf_thres:float|None = None
    ):
        self.max_age = max_age if max_age is not None else 0
        self.min_hits = min_hits if min_hits is not None else 1
        self.iou_thres = iou_thres if iou_thres is not None else -np.inf
        self.conf_thres = conf_thres if conf_thres is not None else 0.
        self.frame_count = 0
        self.trackers: list[TrackingObject] = []

    def update(self, detections: list[TrackingObject]):
        """
        Update the tracker state for one frame.

        """
        self.frame_count += 1

        # DEBUG:
        # TODO: Create an empty matrix in all black

        def _to_int_bbox(bbox: np.ndarray) -> tuple[int, int, int, int]:
            arr = np.asarray(bbox).reshape(-1)
            x1, y1, x2, y2 = arr[:4]
            return int(x1), int(y1), int(x2), int(y2)

        def _draw_bbox(img: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], label: str | None = None):
            x1, y1, x2, y2 = _to_int_bbox(bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        bboxes_for_canvas: list[np.ndarray] = []
        for det in detections:
            if det.bbox is not None:
                bboxes_for_canvas.append(np.asarray(det.bbox))

        for idx, trk in reversed(list(enumerate(self.trackers))):
            bbox = trk.predict()
            if np.any(np.isnan(bbox)):
                self.trackers.pop(idx)
                continue
            self.trackers[idx].bbox = bbox
            bboxes_for_canvas.append(np.asarray(bbox))

        if len(bboxes_for_canvas):
            max_x = max(int(np.max(bbox[:4:2])) for bbox in bboxes_for_canvas)
            max_y = max(int(np.max(bbox[1:4:2])) for bbox in bboxes_for_canvas)
            width = max(1, max_x + 10)
            height = max(1, max_y + 10)
            image = np.zeros((height, width, 3), dtype=np.uint8)

            for det in detections:
                if det.bbox is not None:
                    _draw_bbox(image, det.bbox, (80, 220, 255), "det")

            for trk in self.trackers:
                if trk.bbox is not None:
                    label = f"id:{trk.id}" if trk.id is not None else "trk"
                    _draw_bbox(image, trk.bbox, (255, 120, 60), label)

            cv2.imshow("ByteTrack Debug", image)
            cv2.waitKey(1)

        high_score_detections = [det for det in detections if det.score >= self.conf_thres]
        low_score_detections  = [det for det in detections if det.score <  self.conf_thres]

        matched_high, unmatched_det_high, unmatched_trk_high = assign(
            high_score_detections, self.trackers, self.iou_thres
        )

        for det_idx, trk_idx in matched_high:
            self.trackers[trk_idx].update(high_score_detections[det_idx])

        if len(unmatched_trk_high) > 0 and len(low_score_detections) > 0:
            remaining_trackers = [self.trackers[i] for i in unmatched_trk_high]

            matched_low, _, _ = assign(
                low_score_detections, remaining_trackers, self.iou_thres
            )

            for det_idx, local_trk_idx in matched_low:
                trk_idx = unmatched_trk_high[local_trk_idx]  # map local index -> global index
                self.trackers[trk_idx].update(low_score_detections[det_idx])

        for det_idx in unmatched_det_high:
            trk = high_score_detections[det_idx]
            trk.id = ByteTrack.count
            ByteTrack.count += 1
            self.trackers.append(trk)

        self.trackers = [t for t in self.trackers if t._time_since_update <= self.max_age]

        outputs = []
        for t in self.trackers:
            if t._hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                outputs.append(t)

        return outputs
