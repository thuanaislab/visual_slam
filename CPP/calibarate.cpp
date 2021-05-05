#include <iostream>
#include <random>
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
std::string trajectory_file = "/home/thuan/Downloads/rgbd_dataset_freiburg2_desk/groundtruth.txt";
//std::string trajectory_file = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1/seq_01_poses";
//std::string image_path = "/home/thuan/Desktop/Paper2/augmented_idea/aalto_dataset_git/kitchen1";
std::string image_path = "/home/thuan/Downloads/rgbd_dataset_freiburg2_desk/rgb/";

Point2d pixel2cam(const Point2d &p, const Mat &K);
void ramdom_num(double &x, double &y, double &z);
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
void verify_K(Mat E, std::vector<KeyPoint> keypoints_1,
              std::vector<KeyPoint> keypoints_2,
              std::vector<DMatch> matches, 
              cv::Point2d principal_point,
              double focal_length
                          );
double fRand(double fMin, double fMax);
double cost(Mat E);
int main(){
	std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
	std::vector<std::string> path;
	std::ifstream fin(trajectory_file);
	std::string end_type = ".png";

	int list_checking[] = {0,3};

	if (!fin){
		std::cout << "can not find the trajectory_file"<< std::endl;
		return 1;
	}
	int tt = 0;
	while (!fin.eof()){
		std::string tmp_path;
		double tx, ty, tz, qx, qy, qz, qw;
		fin >> tmp_path >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
		// if (tt > 10){
	 //    	//std::cout << tx << ty << tz << qx << qy << qz << qw <<  "\n";
	 //    	break;
  //   	}
		if ((tt%29) == 0){
	    	Quaterniond q(qw, qx, qy, qz);
	    	q.normalize();
			Eigen::Isometry3d Twr(q);
			Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
			poses.push_back(Twr);
			path.push_back(tmp_path);
		}
		tt = tt + 1;
		//break;
	}
	

	Isometry3d rel_pose04 = poses[list_checking[0]] * poses[list_checking[1]].inverse();

	//Eigen::Matrix t = rel_pose04.translation();

	//cout << "eigen R: " << rel_pose04.rotation() << endl;
	//cout << "translation: " << rel_pose04.translation() << endl;
	//cout << rel_pose04.translation()(1) << endl;

	cv::Mat t_x = (
		cv::Mat_<double>(3,3) << 0, -rel_pose04.translation()(2),
		 rel_pose04.translation()(1),
      rel_pose04.translation()(2), 0, -rel_pose04.translation()(0),
      -rel_pose04.translation()(1), rel_pose04.translation()(0), 0);

	//cout << "t_x:  \n" << t_x << endl;
	cv::Mat R;
	Eigen2cvMat(rel_pose04, R);

	//cout << "R: \n" <<R << endl;
	cv:: Mat E = t_x*R;
	cout << "True cost :" << cost(E) << endl;

	//cout << "E = t_x*R: \n"<< E << endl;
	//--------------------------------------------------------
	//--------------------------------------------------------
	// load image and compute the keypoints
	// cv::Mat img_1 = cv::imread(image_path + path[list_checking[0]] + end_type, CV_LOAD_IMAGE_COLOR);
	// cv::Mat img_2 = cv::imread(image_path + path[list_checking[1]] + end_type, CV_LOAD_IMAGE_COLOR);

	// cout << image_path + path[list_checking[0]] + end_type << endl;

	// assert(img_1.data && img_2.data && "Can not load images!");

	// std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	// std::vector<cv::DMatch> matches;
	// find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	 
	// cv::Mat dst, re_dst;
	// drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, dst, Scalar::all(-1),
	//                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	// resize(dst, re_dst, cv::Size(), 0.25, 0.25);
	// cv::imshow("matches", dst);
	// waitKey();
	
	// Point2d principal_point(960, 540);
	// double focal_length = 500;
	// // verify_K(E ,keypoints_1, keypoints_2, matches, principal_point, focal_length);
	// cout << "size " << poses.size() << endl;
	DrawTrajectory(poses);
	// ___________________________________________________________________



	return 0;
}
double cost(Mat E){
	Mat e,v;
	cv::eigen(E.t()*E,e,v); 
	return (1 - (e.at<double>(1)/e.at<double>(0)));
}


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
void ramdom_num(double init ,double &x, double y, double z, int type) {
	//x = x + fRand(y, z);
	if (type == 1){
		while (true){
		x = x + fRand(y, z);
		if ((x < 0) || (x > 6000)) {
			x = init;
		}
		else {
			break;
		}
	}}
	if (type == 2){
		while (true){
		x = x + fRand(y, z);
		if ((x < (init + y)) || (x > (init + z))) {
			x = init;
		}
		else {
			break;
		}
	}
	}
}

void verify_K(Mat E, std::vector<KeyPoint> keypoints_1,
              std::vector<KeyPoint> keypoints_2,
              std::vector<DMatch> matches, 
              cv::Point2d principal_point,
              double focal_length
                          ){
	vector<Point2f> points1;
	vector<Point2f> points2;

	for (int i = 0; i < (int) matches.size(); i++) {
	points1.push_back(keypoints_1[matches[i].queryIdx].pt);
	points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	Mat F, KTFK, K_i; 
	F = findFundamentalMat(points1, points2, CV_FM_8POINT);
	cout << "F: " << F << endl;
	Mat K = (Mat_<double>(3, 3) << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1);
	double error;
	double global_error = 10.0;
	double g_focal, g_x, g_y;
	double int1, int2, int3;
	int1 = focal_length;
	int2 = principal_point.x;
	int3 = principal_point.y;

	while (global_error > 0.01){
		ramdom_num(int1,focal_length, -100.0,100.0, 1);
		ramdom_num(int2,principal_point.x, -10.0,10.0, 2);
		ramdom_num(int3,principal_point.y, -10.0,10.0, 2);

		K = (Mat_<double>(3, 3) << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1);
		E = findEssentialMat(points1, points2, focal_length, principal_point);
		// calculate E from F 
		//E = K.t() * F * K; 
		error = cost(E);
  		if (global_error > error){
  			global_error = error;
  			cout << "Error: " << global_error << endl;
  			g_focal = focal_length;
  			g_x = principal_point.x;
  			g_y = principal_point.y;
  		}
  		//break;
	}
	cout << "g_focal " << g_focal << endl;
	cout << "g_x " << g_x << endl;
	cout << "g_y " << g_y << endl;
}

double Cost_from_matched_points(Mat E, std::vector<KeyPoint> keypoints_1,
              std::vector<KeyPoint> keypoints_2,
              std::vector<DMatch> matches, 
              cv::Point2d principal_point,
              double focal_length){
	double error; 
	double g_focal, g_x, g_y;
	double global_error = 1;
	Mat K = (Mat_<double>(3, 3) << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1);
	for (DMatch m: matches) {
			Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
			//Point2d pt1(keypoints_1[m.queryIdx].pt.x, keypoints_1[m.queryIdx].pt.y);
			Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
			Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
			//Point2d pt2(keypoints_2[m.trainIdx].pt.x, keypoints_2[m.trainIdx].pt.y);
			Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
			Mat d = y2.t() * E * y1;
			error = error + abs(d.at<double>(0));
			cout << "epipolar constraint = " << d << endl;
	  		}
  		error = error/matches.size();
  	return error;
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
	cout << "E(1,0)/F(1,0): " <<E.at<double>(1,0)/fundamental_matrix.at<double>(1,0) << endl;
	cout << "E(1,1)/F(1,1): " <<E.at<double>(1,1)/fundamental_matrix.at<double>(1,1) << endl;
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


Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
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
