import lap
import numba
import numpy as np

def linear_assignment(cost_matrix):
    """
    Resolve the assignment matrix via LAPJV if available, otherwise fallback to SciPy.
    """
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #


def iou_batch(bb_test, bb_gt):
    """
    Compute the IOU between two bounding box arrays.
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
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
    Compute the IOU between two bounding box arrays using a numba-optimized loop.
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


def convert_bbox_to_z(bbox):
    """
    Convert [x1, y1, x2, y2] to Kalman filter state form.

    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
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
    Convert Kalman filter state to bounding box coordinates.

    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanFilter(object):
    """
    Lightweight Kalman filter with predict/update for constant velocity models.
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
        """
        Run the prediction step using the configured dynamics.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Assimilate a measurement vector z into the filter state.
        """
        z = np.asarray(z).reshape((self.dim_z, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y

        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox, score=1.0, label=-1):
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.score = float(score)
        self.label = int(label) if label is not None else -1

    def update(self,bbox, score=None, label=None):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

        if score is not None:
            self.score = float(score)
        if label is not None:
            self.label = int(label)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if(self.time_since_update>0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detection boxes shaped [N, 4] to tracker boxes shaped [M, 4].

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

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


class Sort(object):
    """
    SORT multi-object tracker exposed as a simple class with update() API.
    """
    # Configure SORT with tracking hyper-parameters.
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    # Update trackers with detection array shaped (N, >=6).
    def update(self, dets=np.empty((0, 6))):
        """
        Params:
        dets - numpy array shaped [num_dets, K] where the first six values are
                [x1, y1, x2, y2, score, class] and any additional columns are ignored.
        Requires: this method must be called once for each frame even with empty detections
                (use np.empty((0, 6)) for frames without detections).
        Returns:
        numpy array shaped [num_tracks, 7] with columns
        [x1, y1, x2, y2, score, class, track_id]

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets = np.asarray(dets, dtype=float)
        if dets.ndim == 1:
            dets = dets[None, :]
        num_rows = dets.shape[0]
        num_cols = dets.shape[1] if dets.ndim == 2 else 0
        if num_cols > 0 and num_cols < 4:
            raise ValueError("Each detection must contain at least [x1, y1, x2, y2].")

        if num_rows == 0:
            boxes = np.empty((0, 4), dtype=float)
            scores = np.empty((0,), dtype=float)
            labels = np.empty((0,), dtype=float)
        else:
            boxes = dets[:, :4]

        if num_cols >= 5:
            scores = dets[:, 4]
        else:
            scores = np.ones(num_rows, dtype=float)

        if num_cols >= 6:
            labels = dets[:, 5]
        else:
            labels = np.full(num_rows, -1, dtype=float)

        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks = np.asarray(trks)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets = associate_detections_to_trackers(
            boxes, trks[:, :4], self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m
            score = scores[det_idx] if len(scores) > 0 else None
            label = labels[det_idx] if len(labels) > 0 else None
            self.trackers[trk_idx].update(boxes[det_idx, :], score=score, label=label)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            score = scores[i] if len(scores) > 0 else 1.0
            label = labels[i] if len(labels) > 0 else -1
            trk = KalmanBoxTracker(boxes[i, :], score=score, label=label)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                track_vector = np.array([d[0], d[1], d[2], d[3], trk.score, trk.label, trk.id+1]).reshape(1, -1)
                ret.append(track_vector) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,7))
