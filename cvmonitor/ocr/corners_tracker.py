import cv2
import numpy as np

class CornersTracker(object):

    # Parameters - don't change unless absolutely sure
    MAX_FEATURES = 500  # Maximal number of features to track
    MIN_MATCHING_POINTS = 100  # Must be less than MAX_FEATURES
    GOOD_MATCH_PERCENT = 0.25  # Minimum percentage of good features
    TRANS_TYPE = 'affine'  # 'affine' (more robust) \ 'homography' (not recommended)
    TRACKING_PERIOD = 5   # time (in seconds) between initializations of the feature-extractor
    PATCH_EXT_RADIUS = (30, 30)  # (cols, rows) - extension radius of the polygon for feature extraction
    LIGHTING_TH = 25  # Maximal difference of the image mean, in order to continue tracking. Otherwise - reinitialize


    def valid_keypoints (self, keypoints, descriptors, col1, row1, col2, row2):
        validMask = []
        for i in range(len(keypoints)):
            validity = keypoints[i].pt[0] >= col1 and keypoints[i].pt[0] <= col2 and keypoints[i].pt[1] >= row1 and keypoints[i].pt[1] <= row2
            validMask.append(validity)
        keypoints = [keypoints[i] for i in range(len(validMask)) if validMask[i]]
        descriptors = descriptors[validMask, :]
        return keypoints, descriptors


    def track_corners(self, img, timeStamp, points):
        points_prev = points

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_mean = img_gray.mean()

        if 'orb' not in self.__dict__:   # initialize tracker
            self.timeStampLast = timeStamp

            # Initialize handles
            self.orb = cv2.ORB_create(CornersTracker.MAX_FEATURES)
            self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

            col1, row1 = (np.min(points, 0)).astype(int) - CornersTracker.PATCH_EXT_RADIUS
            col2, row2 = (np.max(points, 0)).astype(int) + CornersTracker.PATCH_EXT_RADIUS

            keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)
            keypoints, descriptors = self.valid_keypoints(keypoints, descriptors, col1, row1, col2, row2)
            self.keypoints_old = keypoints
            self.descriptors_old = descriptors
            self.old_points = points

        else:  # Tracking
            col1, row1 = (np.min(points, 0)).astype(int) - CornersTracker.PATCH_EXT_RADIUS
            col2, row2 = (np.max(points, 0)).astype(int) + CornersTracker.PATCH_EXT_RADIUS

            keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)
            keypoints, descriptors = self.valid_keypoints(keypoints, descriptors, col1, row1, col2, row2)
            matches = self.matcher.match(self.descriptors_old, descriptors, None)

            # Sort matches by score
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            numGoodMatches = np.maximum(int(len(matches) * CornersTracker.GOOD_MATCH_PERCENT), CornersTracker.MIN_MATCHING_POINTS)
            matches = matches[:numGoodMatches]

            # Extract location of good matches
            keypoints_old_best = np.zeros((len(matches), 2), dtype=np.float32)
            keypoints_best = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                keypoints_old_best[i, :] = self.keypoints_old[match.queryIdx].pt
                keypoints_best[i, :] = keypoints[match.trainIdx].pt

            # Find homography \ affine and apply it on the corners
            if CornersTracker.TRANS_TYPE == 'affine':
                h, mask = cv2.estimateAffinePartial2D(keypoints_old_best, keypoints_best)
                points = np.squeeze(cv2.transform(self.old_points.reshape(-1, 1, 2), h))
            elif CornersTracker.TRANS_TYPE == 'homography':
                h, mask = cv2.findHomography(keypoints_old_best, keypoints_best, cv2.RANSAC)
                points = np.squeeze(cv2.perspectiveTransform(self.old_points.reshape(-1, 1, 2), h))

            img_mean_diff = img_mean - self.prev_img_mean
            # initialize features every set period, to reduce chance of failure
            if (timeStamp - self.timeStampLast) >= CornersTracker.TRACKING_PERIOD or abs(img_mean_diff) >= CornersTracker.LIGHTING_TH:
                if abs(img_mean_diff) >= CornersTracker.LIGHTING_TH:  # In that case, don't trust the tracking
                    points = points_prev

                self.timeStampLast = timeStamp

                col1, row1 = (np.min(points, 0)).astype(int) - CornersTracker.PATCH_EXT_RADIUS
                col2, row2 = (np.max(points, 0)).astype(int) + CornersTracker.PATCH_EXT_RADIUS

                keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)
                keypoints, descriptors = self.valid_keypoints(keypoints, descriptors, col1, row1, col2, row2)
                self.keypoints_old = keypoints
                self.descriptors_old = descriptors
                self.old_points = points

        self.prev_img_mean = img_mean
        return points
