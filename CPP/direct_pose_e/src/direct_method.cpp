#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<double, 4, 1> vec4;
typedef Eigen::Matrix<double, 3, 1> vec3;
typedef Eigen::Matrix<double, 7, 1> vec7;


std::string image_path = "/home/thuan/Desktop/visual_slam/Data/Original/deer_robot/cam0/data/";
std::string trajectory_file = "/home/thuan/Desktop/visual_slam/Data/Augmentation/deer_robot/poses.txt";
std::string depth_path = "/home/thuan/Desktop/visual_slam/Data/Original/deer_robot/depth0/data/";
std::string save_path = "/home/thuan/Desktop/estimate_poses.txt";
std::string augment_image_path = "/home/thuan/Desktop/visual_slam/Data/Augmentation/deer_robot/left_augmentation/";


typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = -600, fy = 600, cx = 320, cy = 240;
//double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
//double baseline = 0.573;
// // paths
// string left_file = "./left.png";
// string disparity_file = "./disparity.png";
// boost::format fmt_others("./%06d.png");    // other files

int image_pairs[] = {520,524};


// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d &T21_) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

void getQuaternion(cv::Mat R, double Q[]);
void convert2Q_t(Eigen::Isometry3d Tw2, vec7 &tmp_pose);
void Isometry3d_2_cvMat(Eigen::Isometry3d T, cv::Mat &R);
void save_file(std::string file_path, vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> new_poses);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);
void Get_Tw2(Sophus::SE3d T21, Eigen::Isometry3d Tw1, Eigen::Isometry3d &Tw2);
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {

    std::vector<std::string> path;
    std::ifstream fin(trajectory_file);
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> new_poses;
    if (!fin){
        std::cout << "can not find the trajectory_file"<< std::endl;
        return 1;
    }

    while (!fin.eof()) {
        std::string tmp_path;
        double tx, ty, tz, qx, qy, qz, qw;
        fin >> tmp_path >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
        Quaterniond q(qw, qx, qy, qz);
        q.normalize();
        Eigen::Isometry3d Twr(q);
        Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
        poses.push_back(Twr);
        path.push_back(tmp_path);
        //cout << tmp_path << endl;
        //break;
    }
    cout << "poses.size() " << poses.size() << endl;
    cout << "path.size() " << path.size() << endl;


    //Isometry3d relative_pose =  poses[image_pairs[1]].inverse() * poses[image_pairs[0]];
    //for (int i1=0; i1 < poses.size(); i1 ++)
    int i1 = 0;
    while (i1 < (poses.size() - 4))
        {

        //i1 = image_pairs[0];

        cv::Mat left_img = cv::imread(image_path + path[i1], 0);
        cv::Mat depth_img = cv::imread(depth_path + path[i1], CV_LOAD_IMAGE_UNCHANGED);
        
        // cv::Mat test_img = cv::imread(image_path + path[image_pairs[0]]);
        // cv::circle(test_img, cv::Point(557, 74), 5, CV_RGB(255, 0, 0), -1, CV_AA);
        // cv::circle(test_img, cv::Point(505, 117), 5, CV_RGB(255, 255, 255), -1, CV_AA);
        // cv::imshow("ok1", test_img);
        // cv::imshow("ok", depth_img);
        // cv::waitKey(0);
        // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
        cv::RNG rng;
        int nPoints = 2000;
        int boarder = 20;
        VecVector2d pixels_ref;
        vector<double> depth_ref;
        

        // generate pixels in ref and load depth data
        for (int i = 0; i < nPoints; i++) {
            int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
            int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
            //ushort disparity = depth_img.at<uchar>(y, x);
            ushort d = depth_img.ptr<unsigned short>(y)[x];
            if (d == 0)   // bad depth
                continue;
            float dd = d/1000.0;
            //double dd = fx * baseline / disparity; // you know this is disparity to depth
            //double depth = fx;
            depth_ref.push_back(dd);
            pixels_ref.push_back(Eigen::Vector2d(x, y));
        }
        // estimates 01~05.png's pose using this information
        Sophus::SE3d T21;

        cv::Mat img = cv::imread(augment_image_path + path[i1], 0);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T21);

        //cout << T21.matrix() << endl;

        Eigen::Isometry3d Tw2; 

        Eigen::Isometry3d Tw1 = poses[i1];

        Get_Tw2(T21, Tw1, Tw2);

        vec7 estimate_vec7;
        convert2Q_t(Tw2, estimate_vec7);
        vec7 true_pose;
        convert2Q_t(Tw1, true_pose);

        //cout << "true_pose \n" << true_pose << endl;
        //cout << "estimate_vec7 \n" << estimate_vec7 << endl;

        i1 = i1 + 2;

        new_poses.push_back(Tw2);
        cout << i1 << endl;
        if (i1 > 780){
            break;
        }
        
    }
    

    cout << "new_poses.size() "<< new_poses.size() <<endl;
    save_file(save_path, new_poses);

    return 0;
}
void save_file(std::string file_path, vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> new_poses){
    ofstream myfile;
    myfile.open(file_path);
    vec7 tmp_pose;
    for (int i = 0; i < new_poses.size(); i++){
        convert2Q_t(new_poses[i], tmp_pose);
        myfile << tmp_pose(0,0) << " " << tmp_pose(1,0) << " " << tmp_pose(2,0) << " " 
        << tmp_pose(3,0) << " " << tmp_pose(4,0) << " " << tmp_pose(5,0)
         << " " << tmp_pose(6,0) << "\n";
    }
    myfile.close();
}


void Get_Tw2(Sophus::SE3d T21, Eigen::Isometry3d Tw1, Eigen::Isometry3d &Tw2){
    cv::Mat R;

    // convert T21 to Eigen type. 
    R = (cv::Mat_<double>(3,3) << T21.matrix()(0,0), T21.matrix()(0,1),
        T21.matrix()(0,2),T21.matrix()(1,0),T21.matrix()(1,1),
        T21.matrix()(1,2),T21.matrix()(2,0),T21.matrix()(2,1),
        T21.matrix()(2,2));
    double Q[] = {0.0,0.0,0.0,0.0};
    getQuaternion(R,Q);

    Eigen::Quaterniond q(Q[3], Q[0], Q[1], Q[2]);
    q.normalize();
    Eigen::Isometry3d T21_e(q);
    T21_e.pretranslate(Eigen::Vector3d(T21.matrix()(0,3), T21.matrix()(1,3), T21.matrix()(2,3)));
    
    Tw2 = (T21_e * Tw1.inverse()).inverse();
}

void convert2Q_t(Eigen::Isometry3d Tw2, vec7 &tmp_pose){

    cv::Mat R; 
    Isometry3d_2_cvMat(Tw2, R);
    double Q[] = {0.0,0.0,0.0,0.0};
    getQuaternion(R,Q);
    tmp_pose << Tw2.translation()(0), Tw2.translation()(1), Tw2.translation()(2), 
        Q[3], Q[0], Q[1], Q[2];

}


void Isometry3d_2_cvMat(Eigen::Isometry3d T, cv::Mat &R){
    R = (cv::Mat_<double>(3, 3) << T.rotation()(0,0), T.rotation()(0,1),
        T.rotation()(0,2),T.rotation()(1,0),T.rotation()(1,1),
        T.rotation()(1,2),T.rotation()(2,0),T.rotation()(2,1),
        T.rotation()(2,2));
}


void getQuaternion(cv::Mat R, double Q[])
{
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
 
    if (trace > 0.0) 
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
        Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
        Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
    } 
    
    else 
    {
        int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0); 
        int j = (i + 1) % 3;  
        int k = (i + 2) % 3;

        double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
        Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
        Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
    }
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            // cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            // cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        // cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    // cout << "T21 = \n" << T21.matrix() << endl;
    // auto t2 = chrono::steady_clock::now();
    // auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "direct method for single layer: " << time_used.count() << endl;

    //plot the projected pixels here
    // cv::Mat img2_show;
    // cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    // VecVector2d projection = jaco_accu.projected_points();
    // for (size_t i = 0; i < px_ref.size(); ++i) {
    //     auto p_ref = px_ref[i];
    //     auto p_cur = projection[i];
    //     if (p_cur[0] > 0 && p_cur[1] > 0) {
    //         cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
    //         cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
    //                  cv::Scalar(0, 250, 0));
    //     }
    // }
    // cv::imshow("current", img2_show);
    // cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
