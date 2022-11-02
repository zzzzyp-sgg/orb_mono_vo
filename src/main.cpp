#include <fstream>
#include <iostream>
#include <chrono>

#include "VisualOdometry.h"

#define MAX_FRAME 1000

int main(int argc, char **argv)
{
    std::ofstream myfile;
    myfile.open("result1_1.txt");

    double scale = 1.0;
    char filename1[200];
    char filenmae2[200];
    sprintf(filename1, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", 0);
    sprintf(filenmae2, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", 1);

    char text[100];
    int fontFace = CV_FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // 读取数据集中的前两个框架
    cv::Mat img_1_c = cv::imread(filename1);
    cv::Mat img_2_c = cv::imread(filenmae2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout << "--(!)Error reading images " << std::endl;
        return -1;
    }

    // 把彩色图像转灰度图像
    cv::Mat img_1, img_2;
    cv::cvtColor(img_1_c, img_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_c, img_2, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kp_1, kp_2;
    std::vector<cv::DMatch> matches;
    cv::Mat curr_R, curr_t;
    
    find_feature_matches(img_1, img_2, kp_1, kp_2, matches);
    pose_eatimation_2d2d(kp_1, kp_2, matches, curr_R, curr_t);

    cv::Mat prevImage = img_2;
    cv::Mat currImage;
    // std::vector<cv::KeyPoint> prev_kp = kp_2;
    // std::vector<cv::KeyPoint> curr_kp;

    char filename[200];
    cv::Mat R_f = curr_R.clone();
    cv::Mat t_f = curr_t.clone();
    std::cout << curr_t.at<double>(0) << " " << curr_t.at<double>(1) << " " << curr_t.at<double>(2) << std::endl;

    auto t1 = std::chrono::steady_clock::now();
    cv::namedWindow( "Road facing camera", cv::WINDOW_AUTOSIZE ); // 这个窗口展示图像
    cv::namedWindow( "Trajectory", cv::WINDOW_AUTOSIZE);          // 这个窗口显示轨迹

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++)
    {
        sprintf(filename, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", numFrame);
        cv::Mat currImage_c = cv::imread(filename);
        cv::cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
        find_feature_matches(prevImage, currImage, kp_1, kp_2, matches);
        pose_eatimation_2d2d(kp_1, kp_2, matches, curr_R, curr_t);
        std::cout << curr_t.at<double>(0) << " " << curr_t.at<double>(1) << " " << curr_t.at<double>(2) << std::endl;

        cv::Mat prevPts(2, kp_1.size(), CV_64F);
        cv::Mat currPts(2, kp_2.size(), CV_64F);

        for (int i = 0; i < kp_1.size(); i++){
            prevPts.at<double>(0, i) = kp_1[i].pt.x;
            prevPts.at<double>(1, i) = kp_1[i].pt.y;
            currPts.at<double>(0, i) = kp_2[i].pt.x;
            currPts.at<double>(1, i) = kp_2[i].pt.y;
        }

        scale = getAbsoluteScale(numFrame);

        if ((scale > 0.1) 
             && (curr_t.at<double>(2) > curr_t.at<double>(0))
             && (curr_t.at<double>(2) > curr_t.at<double>(1)))
        {
            t_f = t_f + scale * (R_f * curr_t);
            R_f = curr_R * R_f;
        }
        // else{
        //     std::cout << "scale below 0.1, or incorrect translation" << std::endl;
        // }

        // lines for printing results
        myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << std::endl;

        prevImage = currImage.clone();

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        // 半径为1,线宽为2,;画出来实心的圆,按照trajectory绘制轨迹
        cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
        // 画出显示的矩形
        cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinate: x = %02fm y = %02fm z = %2fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

        cv::imshow("Road facing camera", currImage_c);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);
    }

    myfile.close();
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    
    std::cout << "Total cost time: " << time_used.count() << "s" << std::endl;

    return 0;
}