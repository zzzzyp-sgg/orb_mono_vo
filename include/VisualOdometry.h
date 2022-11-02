#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <nmmintrin.h>

using namespace cv;
using std::vector;

double getAbsoluteScale(int frame_id);

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    vector<KeyPoint> &keypoints1,
    vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches);

void pose_eatimation_2d2d(
    vector<KeyPoint> keypoints_1, 
    vector<KeyPoint> keypoints_2,
    vector<DMatch> matches,
    Mat &R, Mat &t
);

#endif // VISUAL_ODOMETRY