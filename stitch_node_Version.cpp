#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <future>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <ctime>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
using namespace std::chrono_literals;
namespace fs = std::filesystem;
// ================= Math Helpers =================
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return S;
}
inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-12) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d k = omega / theta;
    Eigen::Matrix3d K = skew(k);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + 
           (1.0 - std::cos(theta)) * (K*K);
}
inline Eigen::Matrix4d se3_exp(const Eigen::Vector3d& omega, const Eigen::Vector3d& v, double dt) {
    Eigen::Matrix3d R = exp_so3(omega*dt);
    Eigen::Vector3d t = v*dt;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}
// Natural sort for filenames
bool natural_sort(const std::string& a, const std::string& b) {
    auto get_number = [](const std::string& s) -> int {
        std::smatch match;
        if (std::regex_search(s, match, std::regex("(\\d+)"))) {
            return std::stoi(match[1].str());
        }
        return 0;
    };
    return get_number(a) < get_number(b);
}

// ================= Point Cloud Data =================
struct PointCloudData {
    Eigen::MatrixXd points;     
    Eigen::VectorXd velocities; 
    Eigen::MatrixXd normals;    
    
    // NEW: Timestamp fields from CSV
    double frame_timestamp_seconds = 0.0;
    double frame_timestamp_nanoseconds = 0.0;
    // Pre-computed for optimization
    Eigen::VectorXd point_norms;
    Eigen::MatrixXd unit_directions;
    void precompute() {
        if (points.rows() == 0) return;
        point_norms = points.rowwise().norm();
        unit_directions.resize(points.rows(), 3);
        for (int i = 0; i < points.rows(); ++i) {
            double norm = point_norms(i);
            if (norm < 1e-12) norm = 1.0;
            unit_directions.row(i) = points.row(i) / norm;
        }
    }
};
// ================= Simple Async Preprocessor =================
class AsyncPreprocessor {
public:
    AsyncPreprocessor() : stop_(false) {
        worker_ = std::thread(&AsyncPreprocessor::processLoop, this);
    }
    ~AsyncPreprocessor() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    std::future<PointCloudData> preprocessAsync(PointCloudData data) {
        // accept by value to allow caller to move if desired
        auto task = std::make_shared<std::packaged_task<PointCloudData()>>(
            [data]() -> PointCloudData {
                // Simulate preprocessing work
                PointCloudData result = data;
                result.precompute();
                std::this_thread::sleep_for(std::chrono::milliseconds(2)); // Simulate work
                return result;
            }
        );   
        std::future<PointCloudData> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if(stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.push([task](){ (*task)(); });
        }
        condition_.notify_one();
        return result;
    }
private:
    void processLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] {
                    return stop_ || !tasks_.empty();
                });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }
    std::thread worker_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

// ================= Doppler ICP Node =================
class DopplerICPStitcher : public rclcpp::Node {
public:
    DopplerICPStitcher() : Node("doppler_icp_stitcher"), 
                          preprocessor_(std::make_unique<AsyncPreprocessor>()),
                          target_frame_time_(0.0),
                          enable_recording_(false) {
        // Parameters with lists for multiple executions
        declare_parameter("frames_directory", "/home/farness/Bureau/outdoor/frames_split");
        declare_parameter("output_csv_path", "/home/zassi/ros2_ws/icp_pose/outdoor.csv");
        declare_parameter("ground_truth_csv", "/home/farness/Bureau/outdoor/groundtruthv1_fixed_CORRECTED.csv");
        // NEW: MCAP recording parameters
        declare_parameter("enable_recording", true);
        declare_parameter("recording_directory", "./recordings");
        // NEW: Parameter lists for multiple executions
        declare_parameter("velocity_threshold", std::vector<double>{0.1});
        declare_parameter("downsample_factor", std::vector<int>{1});  
        declare_parameter("max_iterations", std::vector<int>{10});     
        declare_parameter("icp_tolerance", std::vector<double>{1e-9});    
        declare_parameter("publish_rate", std::vector<double>{10.0});
        declare_parameter("lambda_doppler_start", std::vector<double>{0.0});
        declare_parameter("lambda_doppler_end", std::vector<double>{0.0});
        declare_parameter("lambda_schedule_iters", std::vector<int>{0});
        declare_parameter("frame_dt", std::vector<double>{0.1});
        declare_parameter("t_vl_x", std::vector<double>{0.0});
        declare_parameter("t_vl_y", std::vector<double>{0.0});
        declare_parameter("t_vl_z", std::vector<double>{0.604});
        declare_parameter("reject_outliers", std::vector<bool>{true});
        declare_parameter("outlier_thresh", std::vector<double>{0.001});
        declare_parameter("rejection_min_iters", std::vector<int>{2});
        declare_parameter("geometric_min_iters", std::vector<int>{0});
        declare_parameter("doppler_min_iters", std::vector<int>{2});
        declare_parameter("geometric_k", std::vector<double>{0.2});
        declare_parameter("doppler_k", std::vector<double>{0.3});
        declare_parameter("max_corr_distance", std::vector<double>{0.1});  
        declare_parameter("min_inliers", std::vector<int>{5});
        declare_parameter("last_n_frames", std::vector<int>{15});        
        declare_parameter("use_voxel_filter", std::vector<bool>{false});
        declare_parameter("voxel_size", std::vector<double>{0.0001});
        // NEW: Adaptive normal estimation parameters
        declare_parameter("normal_estimation_mode", std::vector<std::string>{"vertical"}); // "auto", "vertical", "estimated"
        declare_parameter("static_scene_threshold", std::vector<double>{0.5}); // Z-variance threshold for static scene detection

        frames_dir_ = get_parameter("frames_directory").as_string();
        output_csv_path_ = get_parameter("output_csv_path").as_string();
        ground_truth_csv_path_ = get_parameter("ground_truth_csv").as_string();

        // NEW: MCAP recording initialization
        enable_recording_ = get_parameter("enable_recording").as_bool();
        recording_directory_ = get_parameter("recording_directory").as_string();
        
        if (enable_recording_) {
            initialize_recording();
        }
        
        // NEW: Initialize parameter combinations
        initialize_parameter_combinations();
        
        // NEW: Set initial parameters
        set_current_parameters(0);

        // NEW: Enhanced CSV file initialization
        initialize_csv_file();

        // NEW: Initialize logs file
        initialize_logs_file();
        // Load CSV files
        for (auto& entry : fs::directory_iterator(frames_dir_)) {
            std::string ext = entry.path().extension().string();
            std::string filename = entry.path().filename().string();
            if (ext == ".csv" || filename.size() >= 4 && filename.substr(filename.size()-4) == ".csv") {
                frame_files_.push_back(entry.path().string());
                RCLCPP_INFO(get_logger(), "ADDED: %s", filename.c_str());
            }
        }
        std::sort(frame_files_.begin(), frame_files_.end(), natural_sort);
        RCLCPP_INFO(get_logger(), "Loaded %zu CSV frames from %s", frame_files_.size(), frames_dir_.c_str());
        if (!frame_files_.empty()) {
            RCLCPP_INFO(get_logger(), "LOADED %zu FRAMES → FIRST: %s | LAST: %s",
                frame_files_.size(),
                fs::path(frame_files_.front()).filename().c_str(),
                fs::path(frame_files_.back()).filename().c_str());
        }
        if (frame_files_.empty()) {
            RCLCPP_FATAL(get_logger(), "NO FRAME CSVs FOUND! Check path:");
            RCLCPP_FATAL(get_logger(), "Expected: %s", frames_dir_.c_str());
            rclcpp::shutdown();
        }

        // DEBUG: PRINT FIRST AND LAST
        RCLCPP_INFO(get_logger(), "First frame: %s", frame_files_.front().c_str());
        RCLCPP_INFO(get_logger(), "Last  frame: %s", frame_files_.back().c_str());

        // Load ground truth after frame list is known
        load_ground_truth(ground_truth_csv_path_);

        // QoS
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.best_effort();

        // Publishers
        pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("stitched_cloud", qos);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("icp_pose", qos);
        trajectory_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("icp_trajectory", qos);
        lin_acc_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>("linear_acceleration", qos);
        ang_vel_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>("angular_velocity", qos);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        current_pose_ = Eigen::Matrix4d::Identity();
        frame_idx_ = 0;
        next_frame_idx_ = 1;
        previous_frame_set_ = false;
        preprocessing_next_frame_.store(false);
        preprocessed_index_ = std::numeric_limits<size_t>::max();

        // NEW: Initialize timing variables for frame rate control
        last_frame_time_ = std::chrono::steady_clock::now();
        first_frame_processed_ = false;
        target_frame_time_ = frame_dt_; // initialize target frame time from params

        // Start preprocessing next frame immediately if available
        if (frame_files_.size() > 1) {
            start_async_preprocessing(next_frame_idx_);
        }

        // create timer using current publish_rate_
        timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / publish_rate_),
            std::bind(&DopplerICPStitcher::process_next_frame, this));
    }

    ~DopplerICPStitcher() {
        // Close CSV file 
        if (csv_file_.is_open()) {
            csv_file_.close();
            RCLCPP_INFO(get_logger(), "Enhanced trajectory saved to: %s", excel_filename_.c_str());
        }
        
        // NEW: Close logs file
        if (logs_file_.is_open()) {
            logs_file_.close();
            RCLCPP_INFO(get_logger(), "Execution logs saved to: %s", logs_filename_.c_str());
        }
        
        // NEW: Close MCAP recording
        if (bag_writer_) {
            try {
                bag_writer_->close();
                RCLCPP_INFO(get_logger(), "MCAP recording saved: %s/%s", 
                           recording_directory_.c_str(), recording_filename_.c_str());
            } catch (...) {
                RCLCPP_WARN(get_logger(), "Error closing MCAP writer");
            }
        }
        // // FINAL SUMMARY BLOCK
        // // ├─ if GT + ≥2 poses
        // // ├─ evaluate_kpis()  → fresh numbers
        // // ├─ 5x RCLCPP_INFO → big green banner
        // // └─ 3 lines → paste into Table III
        // if (have_ground_truth_ && trajectory_.size() >= 2) {
            
        //     RCLCPP_INFO(get_logger(), "==================================================");
        //     RCLCPP_INFO(get_logger(), "FINAL GLOBAL KPI SUMMARY (Parameter Set %zu)", current_param_index_);
        //     evaluate_kpis();  // final update
        //     RCLCPP_INFO(get_logger(), "RPE:  %.4f m | %.4f deg", last_RPE_trans_, last_RPE_rot_deg_);
        //     RCLCPP_INFO(get_logger(), "ATE:  %.4f m | %.4f deg | %.2f%%", last_ATE_trans_, last_ATE_rot_deg_, last_ATE_percent_);
        //     RCLCPP_INFO(get_logger(), "L:    %.2f m", last_traj_length_);
        //     RCLCPP_INFO(get_logger(), "==================================================");
        // }
    }

private:
    // NEW: Parameter combination structure
    struct ParameterSet {
        double velocity_threshold;
        int downsample_factor;
        int max_iterations;
        double icp_tolerance;
        double publish_rate;
        double lambda_doppler_start;
        double lambda_doppler_end;
        int lambda_schedule_iters;
        double frame_dt;
        double t_vl_x;
        double t_vl_y;
        double t_vl_z;
        bool reject_outliers;
        double outlier_thresh;
        int rejection_min_iters;
        int geometric_min_iters;
        int doppler_min_iters;
        double geometric_k;
        double doppler_k;
        double max_corr_distance;
        int min_inliers;
        int last_n_frames;
        bool use_voxel_filter;
        double voxel_size;
        
        
        // NEW: Adaptive normal estimation parameters
        std::string normal_estimation_mode;
        double static_scene_threshold;
    };

    // NEW: Frame statistics structure for logs
    struct FrameStats {
        size_t frame_index;
        size_t initial_points;
        size_t filtered_points;
        int iterations_used;
        double processing_time_ms;
        std::string filename;
        size_t parameter_set_index;
    };

    // NEW: Timing control variables
    std::chrono::steady_clock::time_point last_frame_time_;
    bool first_frame_processed_;
    double target_frame_time_; // made non-const so we can update when params change

    // NEW: MCAP recording variables
    std::unique_ptr<rosbag2_cpp::Writer> bag_writer_;
    bool enable_recording_;
    std::string recording_directory_;
    std::string recording_filename_;
    std::string ground_truth_csv_path_;

    // NEW: Initialize MCAP recording
    void initialize_recording() {
        // Create recording directory
        std::filesystem::create_directories(recording_directory_);
        
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_time_t);
        
        std::stringstream filename;
        filename << "doppler_icp_"
                 << std::put_time(&now_tm, "%Y%m%d_%H%M%S")
                 << ".mcap";
        
        recording_filename_ = filename.str();
        std::string full_path = recording_directory_ + "/" + recording_filename_;
        
        try {
            bag_writer_ = std::make_unique<rosbag2_cpp::Writer>();
            
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = full_path;
            storage_options.storage_id = "mcap";
            
            // Don't specify converter options - use defaults
            bag_writer_->open(storage_options);
            
            RCLCPP_INFO(get_logger(), "Started MCAP recording: %s", full_path.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize MCAP recording: %s", e.what());
            RCLCPP_WARN(get_logger(), "Continuing without MCAP recording");
            enable_recording_ = false;
            bag_writer_.reset();
        }
    }

    // NEW: Record message to MCAP
    template<typename T>
    void record_message(const T& message, const std::string& topic_name) {
        if (!enable_recording_ || !bag_writer_) {
            return;
        }
        
        try {
            bag_writer_->write(message, topic_name, now());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to record message on topic %s: %s", 
                        topic_name.c_str(), e.what());
        } catch (...) {
            RCLCPP_ERROR(get_logger(), "Unknown failure trying to record message on topic %s", topic_name.c_str());
        }
    }
    // private:
    // ├─ 6x last_… → fresh KPIs every frame
    // ├─ alignment_transform_ → 4×4 Kabsch fix
    // compute_alignment()
    // ├─ Grab first 5 poses
    // ├─ SVD → rigid align
    // └─ Store → honest ATE forever
    private:
    // Add near other members
        double last_RPE_trans_ = 0.0;
        double last_RPE_rot_deg_ = 0.0;
        double last_ATE_trans_ = 0.0;
        double last_ATE_rot_deg_ = 0.0;
        double last_ATE_percent_ = 0.0;
        double last_traj_length_ = 0.0;
        // NEW: Alignment transform from GT[0] to estimated[0]
    Eigen::Matrix4d alignment_transform_ = Eigen::Matrix4d::Identity();
    void compute_alignment() {
        if (trajectory_.empty() || ground_truth_poses_.empty()) return;

        // choose number of poses to use for alignment (use min(5, count))
        size_t useN = std::min<size_t>({1, trajectory_.size(), ground_truth_poses_.size()});
        if (useN < 1) useN = 1;

        // Collect corresponding translation vectors
        Eigen::MatrixXd P(3, useN), Q(3, useN);
        for (size_t i = 0; i < useN; ++i) {
            P.col(i) = trajectory_[i].block<3,1>(0,3);
            Q.col(i) = ground_truth_poses_[i].block<3,1>(0,3);
        }

        // centroids
        Eigen::Vector3d p_centroid = P.rowwise().mean();
        Eigen::Vector3d q_centroid = Q.rowwise().mean();

        // demeaned matrices
        Eigen::MatrixXd P0 = P.colwise() - p_centroid;
        Eigen::MatrixXd Q0 = Q.colwise() - q_centroid;

        // compute covariance
        Eigen::Matrix3d H = P0 * Q0.transpose();

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d R = V * U.transpose();

        // Fix possible reflection
        if (R.determinant() < 0) {
            R.col(2) *= -1;
        }

        Eigen::Vector3d t = q_centroid - R * p_centroid;

        alignment_transform_ = Eigen::Matrix4d::Identity();
        alignment_transform_.block<3,3>(0,0) = R;
        alignment_transform_.block<3,1>(0,3) = t;

        double ang_deg = rot_angle(R) * 180.0 / M_PI;
        double trans_norm = t.norm();

        RCLCPP_INFO(get_logger(), "Computed alignment (Kabsch) using %zu poses: rotation=%.6f deg, translation=%.6f m", useN, ang_deg, trans_norm);
    }



    //  step by step

    // if (gt_times.empty() || ground_truth_poses_.empty()) return std::numeric_limits<size_t>::max();
    // If either vector is empty, the function cannot find a match. It returns the sentinel max() value.
    // double ft = frame_seconds + frame_nanoseconds * 1e-9;
    // Combines the two parts of the frame time into a single double representing seconds: frame_seconds (integral seconds) + frame_nanoseconds converted to seconds (nanoseconds * 1e-9).
    // size_t best = 0;
    // Start by assuming the first ground-truth sample (index 0) is the best match.
    // double best_dt = std::abs(gt_times[0] - ft);
    // Compute the absolute time difference between the frame time and the first ground-truth time.
    // for (size_t i = 1; i < gt_times.size(); ++i) { ... }
    // Iterate through remaining gt_times, compute the absolute difference dt between gt_times[i] and ft.
    // If dt < best_dt, update best_dt and best index.
    // return best;
    // After scanning all timestamps, return the index with the smallest absolute time difference
    
    // ---------- KPI evaluation ----------
    std::vector<Eigen::Matrix4d>   ground_truth_poses_;
    bool                          have_ground_truth_ = false;
    std::vector<double>           gt_times; 
    size_t find_best_gt_index_for_frame(double frame_seconds, double frame_nanoseconds) {
        if (gt_times.empty() || ground_truth_poses_.empty()) return std::numeric_limits<size_t>::max();
        double ft = frame_seconds + frame_nanoseconds * 1e-9;
        size_t best = 0;
        double best_dt = std::abs(gt_times[0] - ft);
        for (size_t i = 1; i < gt_times.size(); ++i) {
            double dt = std::abs(gt_times[i] - ft);
            if (dt < best_dt) { best_dt = dt; best = i; }
        }
        return best;
    }
    
    // load_ground_truth():
    // ┌─ Open CSV
    // ├─ Read header
    // ├─ Auto-detect 8 columns 
    // ├─ Fallback: guess columns by position
    // ├─ For every row:
    // │   → split, trim, stod
    // │   → skip NaN / short lines
    // │   → build Quaternion → 4×4 pose
    // ├─ If poses > 1000 m → assume mm → ÷1000
    // ├─ Fill two vectors:
    // │     ground_truth_poses_  ← N × 4×4 matrices
    // │     gt_times             ← N timestamps
    // └─ Print first 3 poses 
    void load_ground_truth(const std::string& csv_path) {
        std::ifstream in(csv_path);
        if (!in.is_open()) {
            RCLCPP_WARN(get_logger(), "Ground-truth CSV not found: %s", csv_path.c_str());
            have_ground_truth_ = false;
            return;
        }
        ground_truth_poses_.clear();
        gt_times.clear(); 
        std::string header_line;
        if (!std::getline(in, header_line)) {
            RCLCPP_WARN(get_logger(), "Empty GT file or missing header: %s", csv_path.c_str());
            have_ground_truth_ = false;
            return;
        }
        // parse header tokens
        std::vector<std::string> headers;
        {
            std::stringstream ss(header_line);
            std::string t;
            while (std::getline(ss, t, ',')) {
                t.erase(0, t.find_first_not_of(" \t\r\n"));
                t.erase(t.find_last_not_of(" \t\r\n") + 1);
                headers.push_back(t);
            }
        }

        // helper: find column index by candidates
        auto find_idx = [&](const std::vector<std::string>& candidates)->int {
            for (size_t i = 0; i < headers.size(); ++i) {
                for (auto &c : candidates) {
                    if (headers[i] == c) return static_cast<int>(i);
                }
            }
            return -1;
        };

        int idx_time = find_idx({"timestamp","time","t","ros_time","header_stamp","sec"});
        int idx_x = find_idx({"x","pos_x","px","tx","position_x"});
        int idx_y = find_idx({"y","pos_y","py","ty","position_y"});
        int idx_z = find_idx({"z","pos_z","pz","tz","position_z"});
        int idx_qx = find_idx({"qx","orient_x","qx_wrt","orientation_x"});
        int idx_qy = find_idx({"qy","orient_y","qy_wrt","orientation_y"});
        int idx_qz = find_idx({"qz","orient_z","qz_wrt","orientation_z"});
        int idx_qw = find_idx({"qw","orient_w","qw_wrt","orientation_w"});
        bool used_fallback = false;
        if (idx_x < 0 || idx_y < 0 || idx_z < 0 || idx_qx < 0 || idx_qy < 0 || idx_qz < 0 || idx_qw < 0) {
            used_fallback = true;
            RCLCPP_WARN(get_logger(), "GT header did not contain expected names; falling back to guessed indices. Header: %s", header_line.c_str());
        }
        std::string line;
        size_t line_idx = 0;
        std::vector<Eigen::Matrix4d> tmp_gt;
        std::vector<double> tmp_times;
        while (std::getline(in, line)) {
            line_idx++;
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<std::string> tok;
            std::string field;
            while (std::getline(ss, field, ',')) {
                field.erase(0, field.find_first_not_of(" \t\r\n"));
                field.erase(field.find_last_not_of(" \t\r\n") + 1);
                tok.push_back(field);
            }
            if (tok.empty()) continue;

            try {
                int tx_i = idx_x, ty_i = idx_y, tz_i = idx_z, tqx_i = idx_qx, tqy_i = idx_qy, tqz_i = idx_qz, tqw_i = idx_qw;
                double time_val = 0.0;
                if (used_fallback) {
                    if (tok.size() >= 8) {
                        tx_i = 1; ty_i = 2; tz_i = 3; tqx_i = 4; tqy_i = 5; tqz_i = 6; tqw_i = 7;
                        if (idx_time < 0) idx_time = 0;
                    } else if (tok.size() >= 7) {
                        tx_i = 0; ty_i = 1; tz_i = 2; tqx_i = 3; tqy_i = 4; tqz_i = 5; tqw_i = 6;
                    } else {
                        RCLCPP_WARN(get_logger(), "GT line %zu has unexpected token count (%zu). Skipping.", line_idx, tok.size());
                        continue;
                    }
                } else {
                    if (idx_time >= 0 && idx_time < static_cast<int>(tok.size())) {
                        try { time_val = std::stod(tok[idx_time]); } catch(...) { time_val = 0.0; }
                    }
                }

                if (tx_i < 0 || ty_i < 0 || tz_i < 0 || tqx_i < 0 || tqy_i < 0 || tqz_i < 0 || tqw_i < 0) {
                    RCLCPP_WARN(get_logger(), "Skipping GT line %zu: missing indices after fallback", line_idx);
                    continue;
                }
                if (static_cast<size_t>(std::max({tx_i,ty_i,tz_i,tqx_i,tqy_i,tqz_i,tqw_i})) >= tok.size()) {
                    RCLCPP_DEBUG(get_logger(), "Skipping GT line %zu: token index out of range (tokens=%zu)", line_idx, tok.size());
                    continue;
                }

                double tx = std::stod(tok[tx_i]);
                double ty = std::stod(tok[ty_i]);
                double tz = std::stod(tok[tz_i]);
                double qx = std::stod(tok[tqx_i]);
                double qy = std::stod(tok[tqy_i]);
                double qz = std::stod(tok[tqz_i]);
                double qw = std::stod(tok[tqw_i]);

                if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz)) {
                    RCLCPP_WARN(get_logger(), "Non-finite GT translation on line %zu: tx=%g ty=%g tz=%g - skipping", line_idx, tx, ty, tz);
                    continue;
                }

                Eigen::Quaterniond q(qw, qx, qy, qz);
                if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || !std::isfinite(q.w())) {
                    RCLCPP_WARN(get_logger(), "Non-finite GT quaternion on line %zu - skipping", line_idx);
                    continue;
                }
                q.normalize();
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3,3>(0,0) = q.toRotationMatrix();
                T(0,3) = tx; T(1,3) = ty; T(2,3) = tz;

                tmp_gt.push_back(T);
                tmp_times.push_back(time_val);
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Exception parsing GT line %zu: %s. Raw: %s", line_idx, e.what(), line.c_str());
                continue;
            }
        }

        if (tmp_gt.empty()) {
            RCLCPP_WARN(get_logger(), "No valid ground-truth poses parsed from %s", csv_path.c_str());
            have_ground_truth_ = false;
            return;
        }

        if (ground_truth_poses_.size() != frame_files_.size()) {
            RCLCPP_ERROR(get_logger(), 
                "CRITICAL: Frame count mismatch! CSVs: %zu, GT poses: %zu", 
                frame_files_.size(), ground_truth_poses_.size());
        } else {
            RCLCPP_INFO(get_logger(), "Perfect frame alignment: %zu CSVs = %zu GT poses", 
                        frame_files_.size(), ground_truth_poses_.size());
        }
        double mean_norm = 0.0;
        for (const auto &T : tmp_gt) mean_norm += T.block<3,1>(0,3).norm();
        mean_norm /= tmp_gt.size();
        bool scaled = false;
        if (mean_norm > 1000.0) {
            
            RCLCPP_WARN(get_logger(), "Mean GT translation magnitude = %.3f > 1000; assuming units in mm and scaling to meters.", mean_norm);
            for (auto &T : tmp_gt) {
                T(0,3) *= 1e-3;
                T(1,3) *= 1e-3;
                T(2,3) *= 1e-3;
            }
            scaled = true;
        }

        // Move to member variables
        ground_truth_poses_ = tmp_gt;
        gt_times = tmp_times;

        // update have_ground_truth_ based on count vs frames
        have_ground_truth_ = (!ground_truth_poses_.empty() && ground_truth_poses_.size() >= frame_files_.size());
        RCLCPP_INFO(get_logger(), "Loaded %zu ground-truth poses (scaled=%s) (have GT = %s)",
                    ground_truth_poses_.size(), scaled ? "YES" : "NO",
                    have_ground_truth_ ? "YES" : "NO");

        // Log first few poses for diagnostics
        size_t showN = std::min<size_t>(3, ground_truth_poses_.size());
        for (size_t i = 0; i < showN; ++i) {
            auto &T = ground_truth_poses_[i];
            RCLCPP_INFO(get_logger(), "GT[%zu] t = [%.6f, %.6f, %.6f]", i, T(0,3), T(1,3), T(2,3));
        }
    }

    // trans(T)      → [tx,ty,tz]
    // rot(T)        → 3×3 rotation
    // rot_angle(R)  → θ in [0,π] radians
    // rigid_body_error(est,gt) → GT⁻¹ S est
    // ---------------------------------------------------------------------
    // Extract translation and rotation
    inline Eigen::Vector3d trans(const Eigen::Matrix4d& T) { return T.block<3,1>(0,3); }
    inline Eigen::Matrix3d rot(const Eigen::Matrix4d& T)   { return T.block<3,3>(0,0); }

    // Rotation angle in radians: )
    inline double rot_angle(const Eigen::Matrix3d& R) {
        // Use the numerical-stable formula: angle = atan2( ||R - R^T||/2, (trace(R)-1)/2 )
        double tr = R.trace();
        double cos_arg = (tr - 1.0) * 0.5;
        cos_arg = std::clamp(cos_arg, -1.0, 1.0);
        // This yields [0, pi]
        return std::acos(cos_arg);
    }
    inline Eigen::Matrix4d rigid_body_error(const Eigen::Matrix4d& estimated, const Eigen::Matrix4d& truth) {
        return truth.inverse() * alignment_transform_ * estimated;
    }
    
    // ================= KPI =================
    // evaluate_kpis()
    // ├─ Skip if no GT
    // ├─ N = min(est,gt)
    // ├─ RPE = avg local jump error
    // ├─ ATE = avg global error 
    // ├─ L  = GT path length
    // ├─ %  = 100×ATE/L
    // ├─ Store 6 numbers → CSV columns AK-AP
    // └─ Print green box → screenshot for paper
        void evaluate_kpis() {
            if (!have_ground_truth_ || trajectory_.size() < 2) {
                last_RPE_trans_ = last_RPE_rot_deg_ = last_ATE_trans_ = last_ATE_rot_deg_ = last_ATE_percent_ = last_traj_length_ = 0.0;
                return;
            }
            
            const size_t N = trajectory_.size();
            const size_t GT = ground_truth_poses_.size();
            
            // Early exit if ground truth doesn't match
            if (GT < N) {
                RCLCPP_WARN(get_logger(), "Not enough ground truth poses (%zu) for trajectory (%zu)", GT, N);
                last_RPE_trans_ = last_RPE_rot_deg_ = last_ATE_trans_ = last_ATE_rot_deg_ = last_ATE_percent_ = last_traj_length_ = 0.0;
                return;
            }
            
            // Validate ground truth translations
            for (size_t i = 0; i < N; ++i) {
                Eigen::Vector3d gt_t = ground_truth_poses_[i].block<3,1>(0,3);
                if (gt_t.norm() > 1e8) {
                    RCLCPP_WARN(get_logger(), "Ground-truth translations look invalid (norm>1e8). Skipping KPI computation.");
                    last_RPE_trans_ = last_RPE_rot_deg_ = last_ATE_trans_ = last_ATE_rot_deg_ = last_ATE_percent_ = last_traj_length_ = 0.0;
                    return;
                }
            }

            // ── ATE Calculation ─────────────────────
            double ate_t = 0.0, ate_r = 0.0;
            for (size_t i = 0; i < N; ++i) {
                Eigen::Matrix4d aligned_est = alignment_transform_ * trajectory_[i];
                Eigen::Matrix4d E = ground_truth_poses_[i].inverse() * aligned_est;
                
                // Translational error
                ate_t += E.block<3,1>(0,3).squaredNorm();
                
                // Rotational error
                Eigen::Matrix3d R = E.block<3,3>(0,0);
                ate_r += std::acos(std::clamp((R.trace()-1)/2, -1.0, 1.0));
            }
            double ATE_trans   = std::sqrt(ate_t / N);
            double ATE_rot_deg = ate_r / N * 180.0 / M_PI;

            // ── Trajectory Length ──────────────────
            double L = 0.0;
            for (size_t i = 0; i + 1 < GT; ++i) {
                Eigen::Vector3d a = ground_truth_poses_[i].block<3,1>(0,3);
                Eigen::Vector3d b = ground_truth_poses_[i+1].block<3,1>(0,3);
                L += (b - a).norm();
            }
            double ATE_percent = L > 1e-6 ? ATE_trans / L * 100.0 : 0.0;

            // ── RPE Calculation ─────────────────────
            per_frame_rpe_trans_.clear();
            per_frame_rpe_rot_deg_.clear();
            double rpe_t = 0.0, rpe_r = 0.0;
            size_t valid_pairs = 0;

            for (size_t i = 0; i + 1 < N; ++i) {
                // Only compute if ground truth available for this pair
                if (i + 1 >= GT) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, 
                                        "Ground truth missing for frame %zu->%zu", i, i+1);
                    continue;
                }

                // Relative motion in estimated trajectory (aligned)
                Eigen::Matrix4d aligned_est_i = alignment_transform_ * trajectory_[i];
                Eigen::Matrix4d aligned_est_i1 = alignment_transform_ * trajectory_[i+1];
                Eigen::Matrix4d E_rel = aligned_est_i.inverse() * aligned_est_i1;
                
                // Relative motion in ground truth trajectory  
                Eigen::Matrix4d G_rel = ground_truth_poses_[i].inverse() * ground_truth_poses_[i+1];
                
                // Error between relative motions
                Eigen::Matrix4d error = G_rel.inverse() * E_rel;
                
                // Calculate THIS FRAME'S error
                double frame_trans_error = error.block<3,1>(0,3).norm();
                Eigen::Matrix3d R_err = error.block<3,3>(0,0);
                double trace = R_err.trace();
                double frame_rot_error = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
                double frame_rot_error_deg = frame_rot_error * 180.0 / M_PI;
                
                // Accumulate
                rpe_t += frame_trans_error;
                rpe_r += frame_rot_error;
                valid_pairs++;
                
                // Store per-frame
                per_frame_rpe_trans_.push_back(frame_trans_error);
                per_frame_rpe_rot_deg_.push_back(frame_rot_error_deg);
            }

            // Compute RPE averages
            double RPE_trans = 0.0, RPE_rot_deg = 0.0;
            if (valid_pairs > 0) {
                RPE_trans = rpe_t / valid_pairs;
                RPE_rot_deg = (rpe_r / valid_pairs) * 180.0 / M_PI;
            }

            // Store results
            last_RPE_trans_ = RPE_trans;
            last_RPE_rot_deg_ = RPE_rot_deg;
            last_ATE_trans_ = ATE_trans;
            last_ATE_rot_deg_ = ATE_rot_deg;
            last_ATE_percent_ = ATE_percent;
            last_traj_length_ = L;

            // Log results
            RCLCPP_INFO(get_logger(), "==================================================");
            RCLCPP_INFO(get_logger(), "=== KPI Summary ===");
            RCLCPP_INFO(get_logger(), "Trajectory length: %.3f m", L);
            RCLCPP_INFO(get_logger(), "ATE: %.6f m (%.3f%%), %.3f deg", ATE_trans, ATE_percent, ATE_rot_deg);
            RCLCPP_INFO(get_logger(), "RPE: %.6f m, %.3f deg (over %zu pairs)", RPE_trans, RPE_rot_deg, valid_pairs);
            RCLCPP_INFO(get_logger(), "==================================================");
      
             
        }
    // NEW: Initialize all parameter combinations
    void initialize_parameter_combinations() {
        auto velocity_thresholds = get_parameter("velocity_threshold").as_double_array();
        auto downsample_factors = get_parameter("downsample_factor").as_integer_array();
        auto max_iterations_list = get_parameter("max_iterations").as_integer_array();
        auto icp_tolerances = get_parameter("icp_tolerance").as_double_array();
        auto publish_rates = get_parameter("publish_rate").as_double_array();
        auto lambda_doppler_starts = get_parameter("lambda_doppler_start").as_double_array();
        auto lambda_doppler_ends = get_parameter("lambda_doppler_end").as_double_array();
        auto lambda_schedule_iters_list = get_parameter("lambda_schedule_iters").as_integer_array();
        auto frame_dts = get_parameter("frame_dt").as_double_array();
        auto t_vl_x_list = get_parameter("t_vl_x").as_double_array();
        auto t_vl_y_list = get_parameter("t_vl_y").as_double_array();
        auto t_vl_z_list = get_parameter("t_vl_z").as_double_array();
        auto reject_outliers_list = get_parameter("reject_outliers").as_bool_array();
        auto outlier_thresh_list = get_parameter("outlier_thresh").as_double_array();
        auto rejection_min_iters_list = get_parameter("rejection_min_iters").as_integer_array();
        auto geometric_min_iters_list = get_parameter("geometric_min_iters").as_integer_array();
        auto doppler_min_iters_list = get_parameter("doppler_min_iters").as_integer_array();
        auto geometric_k_list = get_parameter("geometric_k").as_double_array();
        auto doppler_k_list = get_parameter("doppler_k").as_double_array();
        auto max_corr_distances = get_parameter("max_corr_distance").as_double_array();
        auto min_inliers_list = get_parameter("min_inliers").as_integer_array();
        auto last_n_frames_list = get_parameter("last_n_frames").as_integer_array();
        auto use_voxel_filter_list = get_parameter("use_voxel_filter").as_bool_array();
        auto voxel_sizes = get_parameter("voxel_size").as_double_array();
        
        // NEW: Adaptive normal estimation parameters
        auto normal_estimation_modes = get_parameter("normal_estimation_mode").as_string_array();
        auto static_scene_thresholds = get_parameter("static_scene_threshold").as_double_array();

        // Create all combinations (for simplicity, using the first combination approach)
        // You can extend this to generate all possible combinations if needed
        size_t max_size = std::max({
            velocity_thresholds.size(), downsample_factors.size(), max_iterations_list.size(),
            icp_tolerances.size(), publish_rates.size(), lambda_doppler_starts.size(),
            lambda_doppler_ends.size(), lambda_schedule_iters_list.size(), frame_dts.size(),
            t_vl_x_list.size(), t_vl_y_list.size(), t_vl_z_list.size(), reject_outliers_list.size(),
            outlier_thresh_list.size(), rejection_min_iters_list.size(), geometric_min_iters_list.size(),
            doppler_min_iters_list.size(), geometric_k_list.size(), doppler_k_list.size(),
            max_corr_distances.size(), min_inliers_list.size(), last_n_frames_list.size(),
            use_voxel_filter_list.size(), voxel_sizes.size(),
            normal_estimation_modes.size(), static_scene_thresholds.size()
        });

        parameter_sets_.reserve(max_size);

        for (size_t i = 0; i < max_size; ++i) {
            ParameterSet params;
            params.velocity_threshold = i < velocity_thresholds.size() ? velocity_thresholds[i] : velocity_thresholds[0];
            params.downsample_factor = i < downsample_factors.size() ? downsample_factors[i] : downsample_factors[0];
            params.max_iterations = i < max_iterations_list.size() ? max_iterations_list[i] : max_iterations_list[0];
            params.icp_tolerance = i < icp_tolerances.size() ? icp_tolerances[i] : icp_tolerances[0];
            params.publish_rate = i < publish_rates.size() ? publish_rates[i] : publish_rates[0];
            params.lambda_doppler_start = i < lambda_doppler_starts.size() ? lambda_doppler_starts[i] : lambda_doppler_starts[0];
            params.lambda_doppler_end = i < lambda_doppler_ends.size() ? lambda_doppler_ends[i] : lambda_doppler_ends[0];
            params.lambda_schedule_iters = i < lambda_schedule_iters_list.size() ? lambda_schedule_iters_list[i] : lambda_schedule_iters_list[0];
            params.frame_dt = i < frame_dts.size() ? frame_dts[i] : frame_dts[0];
            params.t_vl_x = i < t_vl_x_list.size() ? t_vl_x_list[i] : t_vl_x_list[0];
            params.t_vl_y = i < t_vl_y_list.size() ? t_vl_y_list[i] : t_vl_y_list[0];
            params.t_vl_z = i < t_vl_z_list.size() ? t_vl_z_list[i] : t_vl_z_list[0];
            params.reject_outliers = i < reject_outliers_list.size() ? reject_outliers_list[i] : reject_outliers_list[0];
            params.outlier_thresh = i < outlier_thresh_list.size() ? outlier_thresh_list[i] : outlier_thresh_list[0];
            params.rejection_min_iters = i < rejection_min_iters_list.size() ? rejection_min_iters_list[i] : rejection_min_iters_list[0];
            params.geometric_min_iters = i < geometric_min_iters_list.size() ? geometric_min_iters_list[i] : geometric_min_iters_list[0];
            params.doppler_min_iters = i < doppler_min_iters_list.size() ? doppler_min_iters_list[i] : doppler_min_iters_list[0];
            params.geometric_k = i < geometric_k_list.size() ? geometric_k_list[i] : geometric_k_list[0];
            params.doppler_k = i < doppler_k_list.size() ? doppler_k_list[i] : doppler_k_list[0];
            params.max_corr_distance = i < max_corr_distances.size() ? max_corr_distances[i] : max_corr_distances[0];
            params.min_inliers = i < min_inliers_list.size() ? min_inliers_list[i] : min_inliers_list[0];
            params.last_n_frames = i < last_n_frames_list.size() ? last_n_frames_list[i] : last_n_frames_list[0];
            params.use_voxel_filter = i < use_voxel_filter_list.size() ? use_voxel_filter_list[i] : use_voxel_filter_list[0];
            params.voxel_size = i < voxel_sizes.size() ? voxel_sizes[i] : voxel_sizes[0];
            
            // NEW: Adaptive normal estimation parameters
            params.normal_estimation_mode = i < normal_estimation_modes.size() ? normal_estimation_modes[i] : normal_estimation_modes[0];
            params.static_scene_threshold = i < static_scene_thresholds.size() ? static_scene_thresholds[i] : static_scene_thresholds[0];

            parameter_sets_.push_back(params);
        }

        RCLCPP_INFO(get_logger(), "Initialized %zu parameter combinations", parameter_sets_.size());
    }

    // NEW: Set current parameters from parameter set
    void set_current_parameters(size_t param_index) {
        if (param_index >= parameter_sets_.size()) {
            RCLCPP_ERROR(get_logger(), "Parameter index out of range: %zu", param_index);
            return;
        }

        const auto& params = parameter_sets_[param_index];
        current_param_index_ = param_index;

        // Set all current parameters
        velocity_threshold_ = params.velocity_threshold;
        downsample_factor_ = params.downsample_factor;
        max_iterations_ = params.max_iterations;
        icp_tolerance_ = params.icp_tolerance;
        publish_rate_ = params.publish_rate;
        lambda_doppler_start_ = params.lambda_doppler_start;
        lambda_doppler_end_ = params.lambda_doppler_end;
        lambda_schedule_iters_ = params.lambda_schedule_iters;
        frame_dt_ = params.frame_dt;
        t_vl_x_ = params.t_vl_x;
        t_vl_y_ = params.t_vl_y;
        t_vl_z_ = params.t_vl_z;
        reject_outliers_ = params.reject_outliers;
        outlier_thresh_ = params.outlier_thresh;
        rejection_min_iters_ = params.rejection_min_iters;
        geometric_min_iters_ = params.geometric_min_iters;
        doppler_min_iters_ = params.doppler_min_iters;
        geometric_k_ = params.geometric_k;
        doppler_k_ = params.doppler_k;
        max_corr_distance_ = params.max_corr_distance;
        min_inliers_ = params.min_inliers;
        last_n_frames_ = params.last_n_frames;
        use_voxel_filter_ = params.use_voxel_filter;
        voxel_size_ = params.voxel_size;
        
        // NEW: Adaptive normal estimation parameters
        normal_estimation_mode_ = params.normal_estimation_mode;
        static_scene_threshold_ = params.static_scene_threshold;

        // update runtime timing policy
        target_frame_time_ = frame_dt_;

        RCLCPP_INFO(get_logger(), "Set parameter combination %zu/%zu", param_index + 1, parameter_sets_.size());
        RCLCPP_INFO(get_logger(), "Normal estimation mode: %s, Static scene threshold: %.3f", 
                   normal_estimation_mode_.c_str(), static_scene_threshold_);

        // Update timer period if publish_rate_ changed: recreate timer_
        if (timer_) {
            timer_->cancel();
        }
        timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / publish_rate_),
            std::bind(&DopplerICPStitcher::process_next_frame, this));
    }

    // NEW: Enhanced CSV file initialization
    void initialize_csv_file() {
        // Create icp_pose directory
        icp_pose_dir_ = "icp_pose";
        std::filesystem::create_directories(icp_pose_dir_);
        
        // Create filename with execution date (no parameter index in filename)
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream filename;
        filename << "execution_time_" << now_time_t << ".csv";
        excel_filename_ = icp_pose_dir_ + "/" + filename.str();
        
        csv_file_.open(excel_filename_, std::ios::out | std::ios::trunc);
        if (!csv_file_.is_open()) {
            RCLCPP_ERROR(get_logger(), "Failed to open output CSV file: %s", excel_filename_.c_str());
            return;
        }
        
        // Enhanced header with all parameters
        csv_file_ << "timestamp,header_frame_id,"
                 << "position_x,position_y,position_z,"
                 << "orientation_x,orientation_y,orientation_z,orientation_w,"
                 << "timestamp,velocity_threshold,downsample_factor,max_iterations,icp_tolerance,"
                 << "lambda_doppler_start,lambda_doppler_end,lambda_schedule_iters,frame_dt,"
                 << "t_vl_x,t_vl_y,t_vl_z,reject_outliers,outlier_thresh,rejection_min_iters,"
                 << "geometric_min_iters,doppler_min_iters,geometric_k,doppler_k,max_corr_distance,"
                 << "min_inliers,last_n_frames,frame_timestamp_seconds,frame_timestamp_nanoseconds,"
                 << "use_voxel_filter,voxel_size,parameter_set_index,"
                 << "normal_estimation_mode,static_scene_threshold,"
                 << "RPE_trans,RPE_rot_deg,ATE_trans,ATE_rot_deg,ATE_trans_percent,traj_length_m\n";
        csv_file_.flush();
        RCLCPP_INFO(get_logger(), "Initialized enhanced trajectory CSV file: %s", excel_filename_.c_str());
        RCLCPP_INFO(get_logger(), "All parameter combinations will be saved to this file");
    }

    // NEW: Initialize logs file for execution statistics
    void initialize_logs_file() {
        // Create logs directory
        logs_dir_ = "logs";
        std::filesystem::create_directories(logs_dir_);
        
        // Create filename with execution date
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream filename;
        filename << "logs_time_execution_" << now_time_t << ".csv";
        logs_filename_ = logs_dir_ + "/" + filename.str();
        logs_file_.open(logs_filename_, std::ios::out | std::ios::trunc);
        if (!logs_file_.is_open()) {
            RCLCPP_ERROR(get_logger(), "Failed to open logs CSV file: %s", logs_filename_.c_str());
            return;
        }
        
        // Header for logs file
        logs_file_ << "frame_index,filename,initial_points,filtered_points,"
                  << "iterations_used,processing_time_ms\n";
        logs_file_.flush();
        
        RCLCPP_INFO(get_logger(), "Initialized execution logs file: %s", logs_filename_.c_str());
    }
    
        // NEW: Save frame statistics to logs file
        void save_frame_stats_to_logs(const FrameStats& stats) {
            if (!logs_file_.is_open()) return;
            
            logs_file_ << std::fixed << std::setprecision(6);
            logs_file_ << stats.frame_index << ","
                    << stats.filename << ","
                    << stats.initial_points << ","
                    << stats.filtered_points << ","
                    << stats.iterations_used << ","
                    << stats.processing_time_ms << "\n";
            
            logs_file_.flush();
        }

    // PER FRAME KPIS
    // save_pose_to_csv()
    // ├─ Skip if file closed
    // ├─ Stamp ROS time
    // ├─ Pull x y z + quat
    // ├─ Match GT by index OR timestamp
    // ├─ Compute per-frame ATE (cm & °)
    // ├─ Write 44 commas → 1  row
    // └─ flush() → instant Excel refresh

    void save_pose_to_csv(size_t frame_idx, const Eigen::Matrix4d& pose, 
                      const PointCloudData& frame_data,
                      double RPE_trans, double RPE_rot, 
                      double ATE_trans, double ATE_rot,
                      double ATE_trans_percent, double L) {
        if (!csv_file_.is_open()) return;

        auto current_time = now();
        
        // Extract position and orientation
        Eigen::Vector3d position = pose.block<3,1>(0,3);
        Eigen::Quaterniond quat(pose.block<3,3>(0,0));
        double ATE_trans_frame = 0.0;
        double ATE_rot_frame_deg = 0.0;

        // ========== ADD PER-FRAME RPE CALCULATION HERE ==========
        double this_frame_rpe_trans = 0.0;
        double this_frame_rpe_rot = 0.0;
        
        if (frame_idx > 0 && (frame_idx - 1) < per_frame_rpe_trans_.size()) {
            // frame_idx-1 because RPE is for transition from previous frame to current
            this_frame_rpe_trans = per_frame_rpe_trans_[frame_idx - 1];
            this_frame_rpe_rot = per_frame_rpe_rot_deg_[frame_idx - 1];
        }
        // For frame 0, RPE is 0 (no previous frame)
        // ========== END PER-FRAME RPE CALCULATION ==========

        if (have_ground_truth_ && frame_idx < ground_truth_poses_.size()) {
            // DIRECT INDEX MATCHING
            Eigen::Matrix4d E = ground_truth_poses_[frame_idx].inverse() * alignment_transform_ * pose;
            ATE_trans_frame = E.block<3,1>(0,3).norm();
            ATE_rot_frame_deg = rot_angle(E.block<3,3>(0,0)) * 180.0 / M_PI;
        }

        // Write enhanced CSV data
        csv_file_ << std::fixed << std::setprecision(6);
        csv_file_ << current_time.seconds() << ","
                << "map,"
                << position.x() << ","
                << position.y() << ","
                << position.z() << ","
                << quat.x() << ","
                << quat.y() << ","
                << quat.z() << ","
                << quat.w() << ","
                << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() << ","
                << velocity_threshold_ << ","
                << downsample_factor_ << ","
                << max_iterations_ << ","
                << icp_tolerance_ << ","
                << lambda_doppler_start_ << ","
                << lambda_doppler_end_ << ","
                << lambda_schedule_iters_ << ","
                << frame_dt_ << ","
                << t_vl_x_ << ","
                << t_vl_y_ << ","
                << t_vl_z_ << ","
                << (reject_outliers_ ? "true" : "false") << ","
                << outlier_thresh_ << ","
                << rejection_min_iters_ << ","
                << geometric_min_iters_ << ","
                << doppler_min_iters_ << ","
                << geometric_k_ << ","
                << doppler_k_ << ","
                << max_corr_distance_ << ","
                << min_inliers_ << ","
                << last_n_frames_ << ","
                << frame_data.frame_timestamp_seconds << ","
                << frame_data.frame_timestamp_nanoseconds << ","
                << (use_voxel_filter_ ? "true" : "false") << ","
                << voxel_size_ << ","
                << current_param_index_ << ","
                << current_normal_mode_ << ","
                << static_scene_threshold_ << ","
                // ========== USE PER-FRAME RPE HERE ==========
                << this_frame_rpe_trans << "," << this_frame_rpe_rot << ","  // ← CHANGED THIS LINE
                // ========== END CHANGE ==========
                << ATE_trans << "," << ATE_rot << ","
                << ATE_trans_percent << "," << L << "\n";
        
        csv_file_.flush();
    }

    // NEW: Detect if scene is static based on point cloud characteristics
    bool is_static_scene(const Eigen::MatrixXd& points) {
        if (points.rows() == 0) return true;
        
        // Calculate variance in z-axis
        double z_mean = 0.0;
        for (int i = 0; i < points.rows(); ++i) {
            z_mean += points(i, 2);
        }
        z_mean /= points.rows();
        
        double z_variance = 0.0;
        for (int i = 0; i < points.rows(); ++i) {
            z_variance += (points(i, 2) - z_mean) * (points(i, 2) - z_mean);
        }
        z_variance /= points.rows();
        
        // Static scenes typically have low z-variance (flat environments)
        // Dynamic scenes have higher z-variance (varied terrain during movement)
        bool is_static = z_variance < static_scene_threshold_;
        
        RCLCPP_DEBUG(get_logger(), "Scene detection - Z variance: %.4f, Static: %s", 
                     z_variance, is_static ? "true" : "false");
        
        return is_static;
    }

    // ================= Async Pipeline Methods =================
    void start_async_preprocessing(size_t frame_idx) {
        if (frame_idx >= frame_files_.size()) return;
        
        preprocessing_next_frame_.store(true);
        // Load frame synchronously but allow caller to move into async (still avoids heavy preprocessing on main thread)
        PointCloudData raw_frame = load_frame(frame_files_[frame_idx]);
        next_frame_future_ = preprocessor_->preprocessAsync(std::move(raw_frame));
        preprocessed_index_ = frame_idx;
        RCLCPP_DEBUG(get_logger(), "Started async preprocessing for frame %zu", frame_idx);
    }

    PointCloudData get_next_preprocessed_frame_for_index(size_t idx) {
        if (!preprocessing_next_frame_.load() || preprocessed_index_ != idx) {
            RCLCPP_WARN(get_logger(), "Requested preprocessed frame not available or index mismatch (requested %zu, available %zu). Loading synchronously", idx, preprocessed_index_);
            return load_frame(frame_files_[idx]);
        }
        
        auto start_wait = std::chrono::high_resolution_clock::now();
        PointCloudData result = next_frame_future_.get();
        preprocessing_next_frame_.store(false);
        preprocessed_index_ = std::numeric_limits<size_t>::max();
        
        auto end_wait = std::chrono::high_resolution_clock::now();
        auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_wait - start_wait);
        RCLCPP_DEBUG(get_logger(), "Retrieved preprocessed frame, wait time: %ld ms", wait_time.count());
        
        return result;
    }

    // ================= Load CSV Frame  =================
    PointCloudData load_frame(const std::string& filename) {
        PointCloudData data;
        std::ifstream in(filename);
        if (!in.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open file: %s", filename.c_str());
            return data;
        }

        std::string line;
        if (!std::getline(in, line)) {
            RCLCPP_ERROR(get_logger(), "Empty file or bad header: %s", filename.c_str());
            return data;
        }

        // Parse header
        std::vector<std::string> headers;
        std::stringstream ss(line);
        std::string h;
        while (std::getline(ss, h, ',')) {
            h.erase(0, h.find_first_not_of(" \t\r\n"));
            h.erase(h.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(h);
        }

        int x_idx = -1, y_idx = -1, z_idx = -1, v_idx = -1;
        int ts_sec_idx = -1, ts_nsec_idx = -1; // NEW: timestamp indices
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == "x") x_idx = i;
            else if (headers[i] == "y") y_idx = i;
            else if (headers[i] == "z") z_idx = i;
            else if (headers[i] == "v_radial" || headers[i] == "radial_vel" || headers[i] == "v") v_idx = i;
            // NEW: Look for timestamp columns
            else if (headers[i] == "frame_timestamp_seconds") ts_sec_idx = i;
            else if (headers[i] == "frame_timestamp_nanoseconds") ts_nsec_idx = i;
        }

        if (x_idx < 0 || y_idx < 0 || z_idx < 0 || v_idx < 0) {
            RCLCPP_ERROR(get_logger(), "Missing required columns in CSV: %s", filename.c_str());
            return data;
        }

        // Reserve space for better performance
        std::vector<Eigen::Vector4d> temp_data;
        temp_data.reserve(10000);
        
        size_t total_points = 0;
        size_t filtered_points = 0; // NEW: Track filtered points
        bool first_valid_row = true; // NEW: For timestamp extraction
        
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::stringstream sl(line);
            std::vector<std::string> row;
            std::string val;
            while (std::getline(sl, val, ',')) row.push_back(val);

            if (row.size() <= static_cast<size_t>(std::max({x_idx, y_idx, z_idx, v_idx}))) continue;

            total_points++;
            double v_radial = 0.0;
            try { 
                v_radial = std::stod(row[v_idx]); 
                
                // NEW: Extract timestamps from first valid row
                if (first_valid_row) {
                    if (ts_sec_idx != -1 && ts_sec_idx < static_cast<int>(row.size())) {
                        try {
                            data.frame_timestamp_seconds = std::stod(row[ts_sec_idx]);
                        } catch (...) { data.frame_timestamp_seconds = 0.0; }
                    }
                    if (ts_nsec_idx != -1 && ts_nsec_idx < static_cast<int>(row.size())) {
                        try {
                            data.frame_timestamp_nanoseconds = std::stod(row[ts_nsec_idx]);
                        } catch (...) { data.frame_timestamp_nanoseconds = 0.0; }
                    }
                    first_valid_row = false;
                }
            } catch (...) { continue; }

            if (std::abs(v_radial) < velocity_threshold_) {
                try {
                    Eigen::Vector4d entry;
                    entry << std::stod(row[x_idx]), std::stod(row[y_idx]), std::stod(row[z_idx]), v_radial;
                    temp_data.push_back(entry);
                    filtered_points++; // NEW: Count filtered points
                } catch (...) {
                    // skip malformed coordinate rows
                    continue;
                }
            }
        }

        size_t N = temp_data.size();
        data.points.resize(N, 3);
        data.velocities.resize(N);
        
        for (size_t i = 0; i < N; ++i) {
            data.points.row(i) = temp_data[i].head<3>().transpose();
            data.velocities(i) = temp_data[i](3);
        }

        // NEW: Store point count statistics
        current_frame_initial_points_ = total_points;
        current_frame_filtered_points_ = filtered_points;

        RCLCPP_DEBUG(get_logger(), "Loaded %s: %zu points, %zu after filtering, timestamps: %.6fs, %.0fns",
                    fs::path(filename).filename().c_str(), total_points, filtered_points,
                    data.frame_timestamp_seconds, data.frame_timestamp_nanoseconds);
        return data;
    }

    // ================= Preprocess Point Cloud  =================
    PointCloudData preprocess_point_cloud(const PointCloudData& input) {
        PointCloudData output;
        
        // Use current parameter values
        bool use_voxel = use_voxel_filter_;
        double voxel_size = voxel_size_;

        auto pcd = std::make_shared<open3d::geometry::PointCloud>();
        pcd->points_.reserve(input.points.rows());
        for (int i = 0; i < input.points.rows(); ++i) {
            pcd->points_.push_back(input.points.row(i).transpose());
        }

        std::shared_ptr<open3d::geometry::PointCloud> pcd_down;
        
        // voxel filter
        if (use_voxel && voxel_size > 0) {
            pcd_down = pcd->VoxelDownSample(voxel_size);
        } else if (downsample_factor_ > 1) {
            pcd_down = pcd->UniformDownSample(downsample_factor_);
        } else {
            pcd_down = pcd;
        }

        if (pcd_down->points_.empty()) return output;

        // NEW: Adaptive normal estimation strategy
        bool use_vertical_normals = false;
        std::string current_normal_mode = normal_estimation_mode_;
        
        // Determine normal estimation strategy
        if (normal_estimation_mode_ == "vertical") {
            use_vertical_normals = true;
            current_normal_mode = "vertical";
        } else if (normal_estimation_mode_ == "estimated") {
            use_vertical_normals = false;
            current_normal_mode = "estimated";
        } else { // "auto" mode - detect scene type
            bool is_static = is_static_scene(input.points);
            use_vertical_normals = is_static;
            current_normal_mode = use_vertical_normals ? "auto_vertical" : "auto_estimated";
        }
        
        // Store the current normal mode for logging
        current_normal_mode_ = current_normal_mode;

        if (use_vertical_normals) {
            RCLCPP_INFO(get_logger(), "Using VERTICAL NORMALS for static scene");
            // Don't estimate normals, we'll set them to vertical
            pcd_down->normals_.resize(pcd_down->points_.size(), Eigen::Vector3d(0.0, 0.0, 1.0));
        } else {
            RCLCPP_INFO(get_logger(), "Using ESTIMATED NORMALS for dynamic scene");
            // Estimate normals with optimized parameters (existing behavior)
            pcd_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(5.0, 20));
        }

        size_t M = pcd_down->points_.size();
        output.points.resize(M, 3);
        output.normals.resize(M, 3);
        output.velocities.resize(M);

        // Map points and normals
        for (size_t i = 0; i < M; ++i) {
            output.points.row(i) = pcd_down->points_[i];
            output.normals.row(i) = pcd_down->normals_[i];
        }

        // Map velocities using nearest neighbor if downsampled
        if (pcd_down->points_.size() < pcd->points_.size()) {
            open3d::geometry::KDTreeFlann kdtree(*pcd);
            for (size_t i = 0; i < M; ++i) {
                std::vector<int> indices(1);
                std::vector<double> dists(1);
                kdtree.SearchKNN(pcd_down->points_[i], 1, indices, dists);
                int idx0 = indices[0];
                output.velocities(i) = (idx0 >= 0 && idx0 < input.velocities.size()) ? input.velocities(idx0) : 0.0;
            }
        } else {
            output.velocities = input.velocities;
        }

        // Copy timestamps
        output.frame_timestamp_seconds = input.frame_timestamp_seconds;
        output.frame_timestamp_nanoseconds = input.frame_timestamp_nanoseconds;

        // Precompute unit directions
        output.precompute();

        return output;
    }

    // ================= Huber Weights  =================
    inline Eigen::VectorXd huber_weights(const Eigen::VectorXd& residuals, double k) {
        const double eps = 1e-12;
        Eigen::VectorXd abs_r = residuals.cwiseAbs();
        return (abs_r.array() <= k).select(
            Eigen::VectorXd::Ones(residuals.size()),
            k / (abs_r.array() + eps)
        );
    }

    // ================= Doppler ICP  =================
    std::pair<Eigen::Matrix4d, double> doppler_icp(const PointCloudData& source, const PointCloudData& target) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Use current parameter values
        int max_iter = max_iterations_;
        double tol = icp_tolerance_;
        double lambda_start = lambda_doppler_start_;
        double lambda_end = lambda_doppler_end_;
        int lambda_iters = lambda_schedule_iters_;
        double dt = frame_dt_;
        Eigen::Vector3d t_vl(t_vl_x_, t_vl_y_, t_vl_z_);
        bool reject_outliers = reject_outliers_;
        double outlier_thresh = outlier_thresh_;
        int rejection_min_iters = rejection_min_iters_;
        int geometric_min_iters = geometric_min_iters_;
        int doppler_min_iters = doppler_min_iters_;
        double geometric_k = geometric_k_;
        double doppler_k = doppler_k_;
        double max_corr_distance = max_corr_distance_;
        int min_inliers = min_inliers_;

        // Preprocess
        PointCloudData src = preprocess_point_cloud(source);
        PointCloudData tgt = preprocess_point_cloud(target);

        if (src.points.rows() == 0 || tgt.points.rows() == 0) {
            RCLCPP_WARN(get_logger(), "Insufficient points; returning identity");
            return {Eigen::Matrix4d::Identity(), std::numeric_limits<double>::infinity()};
        }

        // Build KD-tree for target
        auto tgt_pcd = std::make_shared<open3d::geometry::PointCloud>();
        tgt_pcd->points_.reserve(tgt.points.rows());
        for (int i = 0; i < tgt.points.rows(); ++i) {
            tgt_pcd->points_.push_back(tgt.points.row(i).transpose());
        }
        open3d::geometry::KDTreeFlann kdtree(*tgt_pcd);

        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        double prev_error = std::numeric_limits<double>::infinity();

        // Pre-allocate for performance
        int N = src.points.rows();
        std::vector<int> indices(N);
        std::vector<double> dists(N);
        Eigen::VectorXd r_g(N);
        std::vector<bool> mask(N);

        // Use precomputed unit directions
        const Eigen::MatrixXd& d_unit = src.unit_directions;
        Eigen::MatrixXd r_vecs = src.points.rowwise() + t_vl.transpose();

        int actual_iterations_used = 0; // NEW: Track actual iterations used

        for (int it = 0; it < max_iter; ++it) {
            actual_iterations_used = it + 1; // NEW: Track iteration count
            
            // Lambda scheduling
            double lam = lambda_end;
            if (lambda_iters > 0) {
                double alpha = std::min(1.0, static_cast<double>(it) / lambda_iters);
                lam = lambda_start + (lambda_end - lambda_start) * alpha;
            }

            // Transform source points 
            Eigen::Matrix3d R = transformation.block<3,3>(0,0);
            Eigen::Vector3d t = transformation.block<3,1>(0,3);
            Eigen::MatrixXd src_tf = (R * src.points.transpose()).colwise() + t;
            src_tf.transposeInPlace();

            // Find correspondences 
            int inlier_count = 0;
            for (int i = 0; i < N; ++i) {
                std::vector<int> idx(1);
                std::vector<double> dist(1);
                Eigen::Vector3d query(src_tf(i, 0), src_tf(i, 1), src_tf(i, 2));
                kdtree.SearchKNN(query, 1, idx, dist);
                
                dists[i] = std::sqrt(dist[0]);
                indices[i] = idx[0];
                
                // Geometric residual
                Eigen::Vector3d src_pt(src_tf(i, 0), src_tf(i, 1), src_tf(i, 2));
                Eigen::Vector3d tgt_pt(tgt.points(idx[0], 0), tgt.points(idx[0], 1), tgt.points(idx[0], 2));
                Eigen::Vector3d tgt_norm(tgt.normals(idx[0], 0), tgt.normals(idx[0], 1), tgt.normals(idx[0], 2));
                r_g(i) = (src_pt - tgt_pt).dot(tgt_norm);
                
                // Determine inliers
                bool geom_in = (dists[i] < max_corr_distance);
                bool doppler_in = (!reject_outliers || (it + 1) < rejection_min_iters || 
                                  std::abs(src.velocities(i)) < outlier_thresh);
                mask[i] = geom_in && doppler_in;
                if (mask[i]) inlier_count++;
            }

            if (inlier_count < min_inliers) {
                RCLCPP_WARN(get_logger(), "Insufficient inliers (%d < %d), breaking ICP", inlier_count, min_inliers);
                break;
            }

            // Build linear system 
            Eigen::MatrixXd A(inlier_count * 2, 6);
            Eigen::VectorXd b(inlier_count * 2);
            
            // Compute weights
            Eigen::VectorXd w_g = (it + 1) >= geometric_min_iters ? 
                                  huber_weights(r_g, geometric_k) : 
                                  Eigen::VectorXd::Ones(N);
            
            Eigen::VectorXd w_d = (it + 1) >= doppler_min_iters ? 
                                  huber_weights(src.velocities, doppler_k) : 
                                  Eigen::VectorXd::Ones(N);

            int row_idx = 0;
            
            // Geometric constraints
            for (int j = 0; j < N; ++j) {
                if (!mask[j]) continue;

                Eigen::Vector3d n(tgt.normals(indices[j], 0), tgt.normals(indices[j], 1), tgt.normals(indices[j], 2));
                Eigen::Vector3d p_tf(src_tf(j, 0), src_tf(j, 1), src_tf(j, 2));
                Eigen::Vector3d Jg_omega = -(n.transpose() * skew(p_tf)) * dt;
                Eigen::Vector3d Jg_v = n * dt;
                
                double wg = std::sqrt((1.0 - lam) * w_g(j));
                A.row(row_idx) << Jg_omega.transpose() * wg, Jg_v.transpose() * wg;
                b(row_idx) = -r_g(j) * wg;
                row_idx++;
            }

            // Doppler constraints
            for (int j = 0; j < N; ++j) {
                if (!mask[j]) continue;

                Eigen::Vector3d r_vec_j(r_vecs(j, 0), r_vecs(j, 1), r_vecs(j, 2));
                Eigen::Vector3d d_unit_j(d_unit(j, 0), d_unit(j, 1), d_unit(j, 2));
                Eigen::Vector3d rx_d = r_vec_j.cross(d_unit_j);
                double wd = std::sqrt(lam * w_d(j));
                
                A.row(row_idx) << rx_d.transpose() * wd, d_unit_j.transpose() * wd;
                b(row_idx) = src.velocities(j) * wd;
                row_idx++;
            }

            if (row_idx < 6) {
                RCLCPP_WARN(get_logger(), "Insufficient constraints");
                break;
            }

            // Resize to actual size
            A.conservativeResize(row_idx, 6);
            b.conservativeResize(row_idx);

            // Solve least squares 
            Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
            Eigen::Vector3d omega = x.head<3>();
            Eigen::Vector3d v = x.tail<3>();

            // Update transformation
            Eigen::Matrix4d delta_T = se3_exp(omega, v, dt);
            transformation = delta_T * transformation;

            // Check convergence
            double total_error = 0.0;
            for (int i = 0; i < N; ++i) {
                if (mask[i]) total_error += dists[i] * dists[i];
            }
            total_error = std::sqrt(total_error / inlier_count);

            if (std::abs(prev_error - total_error) < tol) {
                RCLCPP_DEBUG(get_logger(), "Converged at iteration %d", it);
                break;
            }
            prev_error = total_error;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        RCLCPP_DEBUG(get_logger(), "ICP took %ld ms", duration.count());

        // NEW: Store iterations used for logging
        current_frame_iterations_ = actual_iterations_used;

        return {transformation, prev_error};
    }

    // ================= Process Next Frame =================
    void process_next_frame() {
        auto frame_start = std::chrono::steady_clock::now();
        
        // NEW: Wait logic for maintaining target frame time (frame_dt)
        if (first_frame_processed_) {
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                frame_start - last_frame_time_);
            
            if (elapsed.count() < target_frame_time_) {
                auto wait_time = std::chrono::duration<double>(target_frame_time_ - elapsed.count());
                auto wait_start = std::chrono::steady_clock::now();
                
                std::this_thread::sleep_for(wait_time);
                
                auto actual_wait = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - wait_start);
                
                RCLCPP_DEBUG(get_logger(), 
                            "Waited %.3f/%.3f seconds to maintain frame rate", 
                            actual_wait.count(), wait_time.count());
                
                frame_start = std::chrono::steady_clock::now();
            }
        }

        if (frame_idx_ >= frame_files_.size()) {
            RCLCPP_INFO(get_logger(), "All frames processed for parameter set %zu", current_param_index_);
            
            // NEW: Check if there are more parameter combinations to process
            if (current_param_index_ + 1 < parameter_sets_.size()) {
                // Move to next parameter set
                current_param_index_++;
                set_current_parameters(current_param_index_);
                
                // Reset frame processing
                frame_idx_ = 0;
                next_frame_idx_ = 1;
                previous_frame_set_ = false;
                preprocessing_next_frame_.store(false);
                preprocessed_index_ = std::numeric_limits<size_t>::max();
                stitched_frames_.clear();
                trajectory_.clear();
                current_pose_ = Eigen::Matrix4d::Identity();
                
                // NEW: Reset timing for new parameter set
                last_frame_time_ = std::chrono::steady_clock::now();
                first_frame_processed_ = false;
                
                RCLCPP_INFO(get_logger(), "Starting processing with parameter set %zu/%zu", 
                           current_param_index_ + 1, parameter_sets_.size());
                
                // Start preprocessing first frame of new parameter set
                if (frame_files_.size() > 1) {
                    start_async_preprocessing(next_frame_idx_);
                }
                
                return;
            } else {
                RCLCPP_INFO(get_logger(), "All parameter combinations processed");
                timer_->cancel();
            }
            return;
        }

        // Get current frame (try to use preprocessed async frame if it matches the index)
        PointCloudData frame_data;
        if (preprocessing_next_frame_.load() && preprocessed_index_ == frame_idx_) {
            frame_data = get_next_preprocessed_frame_for_index(frame_idx_);
            RCLCPP_DEBUG(get_logger(), "Using preprocessed frame %zu", frame_idx_);
        } else {
            frame_data = load_frame(frame_files_[frame_idx_]);
            RCLCPP_DEBUG(get_logger(), "Loading frame %zu synchronously", frame_idx_);
        }

        // Start preprocessing NEXT frame while current frame processes
        if (frame_idx_ + 1 < frame_files_.size() && !preprocessing_next_frame_.load()) {
            next_frame_idx_ = frame_idx_ + 1;
            start_async_preprocessing(next_frame_idx_);
            RCLCPP_DEBUG(get_logger(), "Started async preprocessing for frame %zu", next_frame_idx_);
        }

        if (frame_data.points.rows() == 0) { 
            RCLCPP_WARN(get_logger(), "Empty frame %zu, skipping", frame_idx_);
            frame_idx_++; 
            return; 
        }

        double dt = frame_dt_;
        Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();

        // ╔══════════════════════════════════════════════════════════╗
        // ║                 DOPPLER-ICP NODE – 30 SEC GUIDE          ║
        // ╟──────────────────────────────────────────────────────────╢
        // ║  FIRST FRAME (frame_idx_ == 0)                           ║
        // ║   pose = Identity                                        ║
        // ║   Kabsch → alignment_transform_                          ║
        // ║   KPIs → 0.00 cm / 0.000°                                ║
        // ║   CSV row 2 → all zeros                                  ║
        // ╟──────────────────────────────────────────────────────────╢
        // ║  EVERY OTHER FRAME                                       ║
        // ║   1. ICP → ΔT (source→target)                            ║
        // ║   2. delta_T = ΔT⁻¹                                      ║
        // ║   3. current_pose_ *= delta_T                            ║
        // ║   4. push cloud + pose → window & trajectory             ║
        // ║   5. evaluate_kpis() → 6 fresh numbers                   ║
        // ║   6. save_pose_to_csv() → new row (AK-AP updated)        ║
        // ║   7. if window > last_n_frames → erase oldest            ║
        // ║   8. lin_acc = Δt / dt                                   ║
        // ║   9. ang_vel = θ / dt                                    ║
        // ║  10. log → “Frame 123: error=0.012 m, 45 ms”             ║
        // ╟──────────────────────────────────────────────────────────╢
        // ║  LIVE IN CSV (open Excel → scroll right)                 ║
        // ║   AK → RPE cm                                            ║
        // ║   AL → RPE °                                             ║
        // ║   AM → ATE cm                                            ║
        // ║   AN → ATE °                                             ║
        // ║   AO → ATE %                                             ║
        // ║   AP → Path L (m)                                        ║
        // ╟──────────────────────────────────────────────────────────╢
        // ║  ONE-LINE PLOT                                           ║
        // ║   pd.read_csv("*.csv").iloc[:,38:44].plot()              ║
        // ╟──────────────────────────────────────────────────────────╢
        // ║  GREEN BOX IN TERMINAL → COPY BOLD → LaTeX               ║
        // ║   \textbf{1.23} & \textbf{4.56}                          ║
        // ╚══════════════════════════════════════════════════════════╝

        if (!previous_frame_set_) {
            current_pose_ = Eigen::Matrix4d::Identity();
            stitched_frames_.push_back({frame_data, current_pose_});
            trajectory_.push_back(current_pose_);
            previous_frame_set_ = true;

            // Compute alignment using first pose and ground truth if available
            compute_alignment();

            // Make sure KPIs are updated before writing (initial KPIs = zeros)
            evaluate_kpis();

            // Save pose with zero KPIs
            save_pose_to_csv(frame_idx_, current_pose_, frame_data,
                            last_RPE_trans_, last_RPE_rot_deg_,
                            last_ATE_trans_, last_ATE_rot_deg_,
                            last_ATE_percent_, last_traj_length_);
                    
            
            
            // NEW: Save frame statistics to logs
            FrameStats stats;
            stats.frame_index = frame_idx_;
            stats.filename = fs::path(frame_files_[frame_idx_]).filename().string();
            stats.parameter_set_index = current_param_index_;
            stats.initial_points = current_frame_initial_points_;
            stats.filtered_points = current_frame_filtered_points_;
            stats.iterations_used = 0; // No ICP for first frame
            auto frame_end = std::chrono::steady_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
            stats.processing_time_ms = frame_duration.count();
            save_frame_stats_to_logs(stats);
            
            RCLCPP_INFO(get_logger(), "Processed initial frame %zu with param set %zu", frame_idx_, current_param_index_);
        } else {
            auto [transform_sp2tp, error] = doppler_icp(previous_frame_, frame_data);
            Eigen::Matrix4d delta_T = transform_sp2tp.inverse();
            current_pose_ = current_pose_ * delta_T;
            stitched_frames_.push_back({frame_data, current_pose_});
            trajectory_.push_back(current_pose_);

            // Update KPIs now that trajectory_ has a new pose
            evaluate_kpis();

            save_pose_to_csv(frame_idx_, current_pose_, frame_data,
                            last_RPE_trans_, last_RPE_rot_deg_,
                            last_ATE_trans_, last_ATE_rot_deg_,
                            last_ATE_percent_, last_traj_length_);

            // Maintain sliding window
            if (last_n_frames_ > 0 && stitched_frames_.size() > static_cast<size_t>(last_n_frames_)) {
                stitched_frames_.erase(stitched_frames_.begin());
            }

            Eigen::Vector3d delta_t = delta_T.block<3,1>(0,3);
            Eigen::Matrix3d delta_R = delta_T.block<3,3>(0,0);
            lin_acc = delta_t / dt;
            Eigen::AngleAxisd aa(delta_R);
            ang_vel = aa.axis() * aa.angle() / dt;

            auto frame_end = std::chrono::steady_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);

            // NEW: Save frame statistics to logs
            FrameStats stats;
            stats.frame_index = frame_idx_;
            stats.filename = fs::path(frame_files_[frame_idx_]).filename().string();
            stats.parameter_set_index = current_param_index_;
            stats.initial_points = current_frame_initial_points_;
            stats.filtered_points = current_frame_filtered_points_;
            stats.iterations_used = current_frame_iterations_;
            stats.processing_time_ms = frame_duration.count();
            save_frame_stats_to_logs(stats);
            
            RCLCPP_INFO(get_logger(), "Frame %zu: error=%.4f, time=%ld ms, param_set=%zu, iterations=%d", 
                       frame_idx_, error, frame_duration.count(), current_param_index_, current_frame_iterations_);
        }

        previous_frame_ = frame_data;
        publish_pointcloud();
        publish_current_pose();
        publish_trajectory();
        publish_lin_acc_ang_vel(lin_acc, ang_vel);
        publish_tf();

        // NEW: Update timing for next frame
        last_frame_time_ = std::chrono::steady_clock::now();
        first_frame_processed_ = true;
        
        frame_idx_++;
        
        
            
    }

    // ================= Publish Functions =================
    void publish_pointcloud() {
        if (stitched_frames_.empty()) return;

        Eigen::Matrix4d current_pose_inv = current_pose_.inverse();
        std::vector<Eigen::Vector3d> all_points;
        all_points.reserve(stitched_frames_.size() * 1000);  // Reserve space

        for (const auto& [frame_data, pose] : stitched_frames_) {
            Eigen::Matrix4d relative_transform = current_pose_inv * pose;
            Eigen::Matrix3d R = relative_transform.block<3,3>(0,0);
            Eigen::Vector3d t = relative_transform.block<3,1>(0,3);

            for (int i = 0; i < frame_data.points.rows(); ++i) {
                Eigen::Vector3d pt = R * frame_data.points.row(i).transpose() + t;
                all_points.push_back(pt);
            }
        }

        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = now();
        msg.header.frame_id = "sensor";
        msg.height = 1;
        msg.width = all_points.size();
        msg.is_dense = false;
        msg.is_bigendian = false;

        sensor_msgs::msg::PointField f;
        f.name = "x"; f.offset = 0; f.datatype = sensor_msgs::msg::PointField::FLOAT32; 
        f.count = 1; msg.fields.push_back(f);
        f.name = "y"; f.offset = 4; msg.fields.push_back(f);
        f.name = "z"; f.offset = 8; msg.fields.push_back(f);

        msg.point_step = 12;
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step);

        uint8_t* ptr = msg.data.data();
        for (const auto& p : all_points) {
            float x = static_cast<float>(p.x());
            float y = static_cast<float>(p.y());
            float z = static_cast<float>(p.z());
            std::memcpy(ptr, &x, 4);
            std::memcpy(ptr + 4, &y, 4);
            std::memcpy(ptr + 8, &z, 4);
            ptr += 12;
        }

        pointcloud_pub_->publish(msg);
        
        // NEW: Record to MCAP
        record_message(msg, "stitched_cloud");
    }

    void publish_current_pose() {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = now();
        pose_msg.header.frame_id = "map";

        Eigen::Vector3d t = current_pose_.block<3,1>(0,3);
        Eigen::Quaterniond q(current_pose_.block<3,3>(0,0));

        pose_msg.pose.position.x = t.x();
        pose_msg.pose.position.y = t.y();
        pose_msg.pose.position.z = t.z();
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        pose_pub_->publish(pose_msg);
        
        // NEW: Record to MCAP
        record_message(pose_msg, "icp_pose");
    }

    void publish_trajectory() {
        if (trajectory_.empty()) return;

        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.stamp = now();
        pose_array.header.frame_id = "map";
        pose_array.poses.reserve(trajectory_.size());

        for (const auto& pose : trajectory_) {
            geometry_msgs::msg::Pose p;
            Eigen::Vector3d t = pose.block<3,1>(0,3);
            Eigen::Quaterniond q(pose.block<3,3>(0,0));

            p.position.x = t.x();
            p.position.y = t.y();
            p.position.z = t.z();
            p.orientation.x = q.x();
            p.orientation.y = q.y();
            p.orientation.z = q.z();
            p.orientation.w = q.w();

            pose_array.poses.push_back(p);
        }

        trajectory_pub_->publish(pose_array);
        
        // NEW: Record to MCAP
        record_message(pose_array, "icp_trajectory");
    }

    void publish_lin_acc_ang_vel(const Eigen::Vector3d& lin_acc, const Eigen::Vector3d& ang_vel) {
        geometry_msgs::msg::Vector3Stamped lin_msg;
        lin_msg.header.stamp = now();
        lin_msg.header.frame_id = "sensor";
        lin_msg.vector.x = lin_acc.x();
        lin_msg.vector.y = lin_acc.y();
        lin_msg.vector.z = lin_acc.z();
        lin_acc_pub_->publish(lin_msg);

        geometry_msgs::msg::Vector3Stamped ang_msg;
        ang_msg.header.stamp = now();
        ang_msg.header.frame_id = "sensor";
        ang_msg.vector.x = ang_vel.x();
        ang_msg.vector.y = ang_vel.y();
        ang_msg.vector.z = ang_vel.z();
        ang_vel_pub_->publish(ang_msg);
        
        // NEW: Record to MCAP
        record_message(lin_msg, "linear_acceleration");
        record_message(ang_msg, "angular_velocity");
    }

    void publish_tf() {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now();
        t.header.frame_id = "map";
        t.child_frame_id = "sensor";
        
        t.transform.translation.x = current_pose_(0, 3);
        t.transform.translation.y = current_pose_(1, 3);
        t.transform.translation.z = current_pose_(2, 3);
        
        Eigen::Quaterniond q(current_pose_.block<3,3>(0,0));
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();
        
        tf_broadcaster_->sendTransform(t);
    }
    
    // ================= Member Variables =================
    std::string frames_dir_;
    std::string output_csv_path_;
    std::vector<double> per_frame_rpe_trans_;
    std::vector<double> per_frame_rpe_rot_deg_;
    
    // NEW: Current parameter values
    double velocity_threshold_;
    int downsample_factor_;
    int max_iterations_;
    double icp_tolerance_;
    double publish_rate_;
    double lambda_doppler_start_;
    double lambda_doppler_end_;
    int lambda_schedule_iters_;
    double frame_dt_;
    double t_vl_x_;
    double t_vl_y_;
    double t_vl_z_;
    bool reject_outliers_;
    double outlier_thresh_;
    int rejection_min_iters_;
    int geometric_min_iters_;
    int doppler_min_iters_;
    double geometric_k_;
    double doppler_k_;
    double max_corr_distance_;
    int min_inliers_;
    int last_n_frames_;
    bool use_voxel_filter_;
    double voxel_size_;
    
    // NEW: Adaptive normal estimation parameters
    std::string normal_estimation_mode_;
    double static_scene_threshold_;
    std::string current_normal_mode_ = "estimated";

    // NEW: Parameter combinations
    std::vector<ParameterSet> parameter_sets_;
    size_t current_param_index_ = 0;

    // NEW: Frame statistics tracking
    size_t current_frame_initial_points_ = 0;
    size_t current_frame_filtered_points_ = 0;
    int current_frame_iterations_ = 0;

    size_t frame_idx_;
    std::vector<std::string> frame_files_;
    std::vector<std::pair<PointCloudData, Eigen::Matrix4d>> stitched_frames_;
    std::vector<Eigen::Matrix4d> trajectory_;
    
    PointCloudData previous_frame_;
    bool previous_frame_set_;
    Eigen::Matrix4d current_pose_;

    std::ofstream csv_file_;
    std::string icp_pose_dir_;
    std::string excel_filename_;

    // NEW: Logs file variables
    std::ofstream logs_file_;
    std::string logs_dir_;
    std::string logs_filename_;

    // ================= Real-time Pipeline Members =================
    std::unique_ptr<AsyncPreprocessor> preprocessor_;
    std::future<PointCloudData> next_frame_future_;
    std::atomic<bool> preprocessing_next_frame_{false};
    size_t next_frame_idx_{1};
    size_t preprocessed_index_{std::numeric_limits<size_t>::max()};

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr lin_acc_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr ang_vel_pub_;
    
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
};

// ================= Main =================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DopplerICPStitcher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}