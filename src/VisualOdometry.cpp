#include <string>

#include "VisualOdometry.h"

double getAbsoluteScale(int frame_id)
{
    std::string line;
    int i = 0;
    std::ifstream myfile("/home/zyp/DATA/KITTI_DATA/dataset/poses/00.txt");   // 真实轨迹的文件地址
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while ((getline(myfile, line)) && (i <= frame_id))
        {
            z_prev = z;
            y_prev = y;
            x_prev = x;
            std::istringstream in(line);
            // [R|t] 每一行的最后一列数据为平移的t的数据
            // 把位移t提取出来
            for (int j = 0; j < 12; j++){
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }
            i++;
        }
        myfile.close();
    }

    else{
        std::cout << "Unable to open file";
        return 0;
    }
    
    // 当前帧的（x，y，z）减去上一帧的（x，y，z）作为真实距离
    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev)
                + (z - z_prev) * (z - z_prev));
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches)
{
    //-- 初始化
    Mat descriptors_1,descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步，检测Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步，根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步，对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步，匹配点对筛选
    double min_dist = 10000, max_dist = 0;  // 求最小距离，初始值可以依据经验设定

    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // printf("--Max dist : %f \n", max_dist);
    // printf("--Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离非常小，设置经验值30作为下限
    for (int i = 0; i < descriptors_1.rows; i++){
        if (match[i].distance <= max(2*min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

void pose_eatimation_2d2d(vector<KeyPoint> keypoints_1, 
                          vector<KeyPoint> keypoints_2,
                          vector<DMatch> matches,
                          Mat &R, Mat &t)
{
    // 相机内参， KITTI
    Mat K = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    // Mat fundamental_matrix;
    // fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    // std::cout << "Fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    //-- 计算本质矩阵
    Point2d principal_point(607.1928, 185.2157);  // 相机光心， KITTI
    double focal_length =718.8560;               // 相机焦距,  KITTI
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    // std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    //-- 从本质矩阵中恢复旋转和平移信息
    // 此函数仅在OpenCV3中使用
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    // std::cout << "R is" << std::endl << R << std::endl;
    // std::cout << "t is" << std::endl << t << std::endl;
}
