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


class KalmanBoxTracker:
    """
    Tracklet that keeps the entire detection info vector alongside the state.
    """
    count = 0

    F = np.array([
        [1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]
    ], dtype=np.float32)

    H = np.array([
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0]
    ], dtype=np.float32)

    def __init__(self, info:np.array):
        """
        Parameters
        ----------
        info : np.ndarray
            Detection vector ``[x1, y1, x2, y2, score, class, ...]`` that
            becomes the initial measurement and metadata payload for the track.
        """
        self.info = info

        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = KalmanBoxTracker.F
        self.kf.H = KalmanBoxTracker.H

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(self.bbox)

        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, info):
        """
        Update the track state using a fresh detection vector.

        The first four entries of `info` must be ``[x1, y1, x2, y2]``; any
        remaining values (score, class, keypoints, etc.) are stored so that
        downstream consumers receive the most recent detector metadata.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(info[:4]))
        self.info = info

    def predict(self):
        """Advance the state vector one frame ahead."""
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if(self.time_since_update>0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    @property
    def get_state(self):
        """Return the current bounding box estimate as ``[x1, y1, x2, y2]``."""
        return convert_x_to_bbox(self.kf.x)
    
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
        bbox = convert_x_to_bbox(self.kf.x).reshape(-1)
        return np.concatenate([bbox, self.info[4:]])
    
    @property
    def get_id(self):
        """
        Return the tracking id.

        Returns
        -------
        int
            The id of the tracker.
        """
        return self.id