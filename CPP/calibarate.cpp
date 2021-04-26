#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;

std::string trajectory_file = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/seq_01_poses";
std::string image_path = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1";

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);
void estimate_K(std::vector<cv::KeyPoint> keypoints_1,
                  std::vector<cv::KeyPoint> keypoints_2,
	              std::vector<cv::DMatch> matches,
	              Mat E, Mat &K);
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
void Eigen2cvMat(Eigen::Isometry3d a_pose, cv::Mat &R);
int main(){
	std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
	std::vector<std::string> path;
	std::ifstream fin(trajectory_file);

	int list_checking[] = {0,3,4,7};

	if (!fin){
		std::cout << "can not find the trajectory_file"<< std::endl;
		return 1;
	}
	int tt = 0;
	while (!fin.eof()){
		std::string tmp_path;
		double tx, ty, tz, qx, qy, qz, qw;
		fin >> tmp_path >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
		if (tt > 10){
	    	//std::cout << tx << ty << tz << qx << qy << qz << qw <<  "\n";
	    	break;
    	}
    	Quaterniond q(qw, qx, qy, qz);
    	q.normalize();
		Eigen::Isometry3d Twr(q);
		Twr.pretranslate(Eigen::Vector3d(tx-24, ty, tz));
		poses.push_back(Twr);
		path.push_back(tmp_path);
		tt = tt + 1;
		//break;
	}
	

	Isometry3d rel_pose04 = poses[list_checking[0]] * poses[list_checking[1]].inverse();

	//Eigen::Matrix t = rel_pose04.translation();

	cout << "eigen R: " << rel_pose04.rotation() << endl;
	cout << "translation: " << rel_pose04.translation() << endl;
	//cout << rel_pose04.translation()(1) << endl;

	cv::Mat t_x = (
		cv::Mat_<double>(3,3) << 0, -rel_pose04.translation()(2),
		 rel_pose04.translation()(1),
      rel_pose04.translation()(2), 0, -rel_pose04.translation()(0),
      -rel_pose04.translation()(1), rel_pose04.translation()(0), 0);

	cout << "t_x:  \n" << t_x << endl;
	cv::Mat R;
	Eigen2cvMat(rel_pose04, R);

	cout << "R: \n" <<R << endl;

	cout << "E = t_x*R: \n"<< t_x*R << endl;
	//--------------------------------------------------------
	//--------------------------------------------------------
	// load image and compute the keypoints
	cv::Mat img_1 = cv::imread(image_path + path[list_checking[0]], CV_LOAD_IMAGE_COLOR);
	cv::Mat img_2 = cv::imread(image_path + path[list_checking[1]], CV_LOAD_IMAGE_COLOR);

	assert(img_1.data && img_2.data && "Can not load images!");

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	std::vector<cv::DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	 
	// cv::Mat dst;
	// drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, dst, Scalar::all(-1),
	//                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	// cv::imshow("matches", dst);
	// waitKey();
	cv::Mat K;
	estimate_K(keypoints_1, keypoints_2, matches, t_x*R, K);

	//DrawTrajectory(poses);
	


	return 0;
}

void estimate_K(std::vector<cv::KeyPoint> keypoints_1,
                  std::vector<cv::KeyPoint> keypoints_2,
	              std::vector<cv::DMatch> matches,
	              Mat E, Mat &K){
	//?? Convert the matching point to the form of vector<Point2f>
  	vector<Point2f> points1;
  	vector<Point2f> points2;

  	for (int i = 0; i < (int) matches.size(); i++) {
    	points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    	points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  	}
  	Mat fundamental_matrix;
  	fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
  	cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;
  	cout << "essential_matrix is " << endl << E << endl;
  	cout << "E(0,0)/F(0,0): " <<E.at<double>(0,0)/fundamental_matrix.at<double>(0,0) << endl;
	cout << "E(0,1)/F(0,1): " <<E.at<double>(0,1)/fundamental_matrix.at<double>(0,1) << endl;
}
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //Oriented FAST 
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);


  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);


  vector<DMatch> match;
  //BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);


  double min_dist = 10000, max_dist = 0;

  
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}



void Eigen2cvMat(Eigen::Isometry3d a_pose, cv::Mat &R){
	R = (Mat_<double>(3, 3) << a_pose.rotation()(0,0), a_pose.rotation()(0,1),
		a_pose.rotation()(0,2),a_pose.rotation()(1,0),a_pose.rotation()(1,1),
		a_pose.rotation()(1,2),a_pose.rotation()(2,0),a_pose.rotation()(2,1),
		a_pose.rotation()(2,2));
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}
