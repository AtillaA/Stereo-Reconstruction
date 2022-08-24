#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;


#ifndef NOCONTRIB
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>

using namespace xfeatures2d;
using namespace ximgproc;
#endif


String unpath(const cv::String& filename) {
    std::size_t pos = filename.find_last_of('/');

    if (pos == String::npos) return filename;
    return filename.substr(pos + 1);
}

String unext(const cv::String& filename) {
    std::size_t pos = filename.find_last_of('.');

    if (pos == 0 || pos == filename.length() -1 || pos == String::npos) return filename;
    return filename.substr(0, pos);
}

const String cli_opts =
    "{help h usage ? |      | Print this message.                                                                     }"
    "{analysis a     |      | Output intermediate results that help analysing how the implementation performs on a dataset.}"
    "{@image1        |<none>| Image 1                                                                                 }"
    "{@image2        |<none>| Image 2                                                                                 }"
//  "{no-rect        |      | Do not rectify the input images before computing the disparity map.                     }"
#ifndef NOCONTRIB
    "{fdetect-descr  |SURF  | Which feature point detection and description algorithm to use. Valid options are \"ORB\", \"SIFT\" and \"SURF\".}"
    "{fdetect        |<none>| Which feature point detection algorithm to use. Valid options are \"ORB\", \"SIFT\" and \"SURF\". }"
    "{fdescr         |<none>| Which feature point description algorithm to use. Valid options are \"ORB\", \"SIFT\" and \"SURF\".}"
#else
    "{fdetect-descr  |SIFT  | Which feature point detection and description algorithm to use. Valid options are \"ORB\" and \"SIFT\".}"
    "{fdetect        |<none>| Which feature point detection algorithm to use. Valid options are \"ORB\" and \"SIFT\". }"
    "{fdescr         |<none>| Which feature point description algorithm to use. Valid options are \"ORB\" and \"SIFT\".}"
#endif
    "{fmatch         |FLANN | Which feature point matching algorithm to use. Valid options are \"BF\" and \"FLANN\".  }"
    "{lowe-ratio     |0.7   | The ratio to use for Lowe's test to check whether a match between two keypoints is good.}"
    "{ransac-thres   |1.0   | RANSAC reprojection threshold.                                                          }"
    "{disp-map       |SGBM  | Which disparity map calculation algorithm to use. Valid options are \"CustomBM\", \"SBM\" and \"SGBM\".}"
#ifndef NOCONTRIB
    "{no-filter      |      | Do not apply post-filtering to the disparity map.                                       }"
#endif
    "{focal-length   |1     | Focal length of the camera                                                              }"
    "{repr-disp-min  |0     | The minimum disparity a pixel must have to be included in the reprojection              }"
    "{repr-disp-max  |100   | The maximum disparity a pixel can have to be included in the reprojection               }"
    "{pointcloud-only|      | Should the program only generate a pointcloud and do not perform mesh reconstruction?   }"
    "{mesh-epsilon   |1     | How close three points in the mesh must be to be connected                              }";

struct Config {
private:
    static std::optional<Config> config;

    Config() = default;
public:
    enum FDetectDescr {
        ORB, SIFT,
#ifndef NOCONTRIB
        SURF
#endif
    };
    enum FMatch { BF, FLANN };
    enum DispMap { CustomBM, SBM, SGBM };


    bool analysis;

    String img1_name;
    String img1_name_no_ext;
    String img2_name;
    String img2_name_no_ext;

    bool no_rect_imgs;

    FDetectDescr feature_detector;
    FDetectDescr feature_descriptor;

    FMatch feature_matcher;
    float lowe_ratio;

    float ransac_thres;

    DispMap disp_map_alg;
#ifndef NOCONTRIB
    bool no_disp_filter;
#endif
    float focal_length;
    float repr_disp_min;
    float repr_disp_max;

    bool pointcloud_only;
    double mesh_epsilon;


    static const Config& get() {
        if (!config) {
            std::cerr << "Error: Config not initialized! Aborting." << std::endl;
            std::exit(1);
        }

        return *config;
    }

    static void init(int argc, char** argv) {
        if (config) return;

        CommandLineParser parser(argc, argv, cli_opts);
        config = Config();


        config->analysis = parser.has("analysis");


        if (parser.has("help")) {
            parser.printMessage();
            std::exit(0);
        }
        if (!parser.check() ||
            !parser.has("@image1") || parser.get<String>(0) == "" ||
            !parser.has("@image2") || parser.get<String>(1) == "") {
            parser.printErrors();
            parser.printMessage();
            std::exit(1);
        }


        config->img1_name = parser.get<String>("@image1");
        config->img1_name_no_ext = unext(unpath(config->img1_name));
        config->img2_name = parser.get<String>("@image2");
        config->img2_name_no_ext = unext(unpath(config->img2_name));


        //config->no_rect_imgs = parser.has("no-rect");


        String fdetect = parser.get<String>("fdetect-descr");
        if (parser.has("fdetect")) {
            fdetect = parser.get<String>("fdetect");
        }
        if (fdetect == "ORB") {
            config->feature_detector = FDetectDescr::ORB;
        }
        else if (fdetect == "SIFT") {
            config->feature_detector = FDetectDescr::SIFT;
        }
#ifndef NOCONTRIB
        else if (fdetect == "SURF") {
            config->feature_detector = FDetectDescr::SURF;
        }
#endif
        else {
            std::cerr << "Warning: Unknown feature detector \"" << fdetect << "\". Using " <<
#ifndef NOCONTRIB
                "\"SURF\"" <<
#else
                "\"SIFT\"" << 
#endif
                " instead" << std::endl;

            config->feature_detector =
#ifndef NOCONTRIB
                FDetectDescr::SURF;
#else
                FDetectDescr::SIFT;
#endif
        }

        String fdescr = parser.get<String>("fdetect-descr");
        if (parser.has("fdescr")) {
            fdescr = parser.get<String>("fdescr");
        }
        if (fdescr == "ORB") {
            config->feature_descriptor = FDetectDescr::ORB;
        }
        else if (fdescr == "SIFT") {
            config->feature_descriptor = FDetectDescr::SIFT;
        }
#ifndef NOCONTRIB
        else if (fdescr == "SURF") {
            config->feature_descriptor = FDetectDescr::SURF;
        }
#endif
        else {
            std::cerr << "Warning: Unknown feature descriptor \"" << fdetect << "\". Using " <<
#ifndef NOCONTRIB
                "\"SURF\"" <<
#else
                "\"SIFT\"" << 
#endif
                " instead" << std::endl;

            config->feature_descriptor =
#ifndef NOCONTRIB
                FDetectDescr::SURF;
#else
                FDetectDescr::SIFT;
#endif
        }


        String fmatch = parser.get<String>("fmatch");
        if (fmatch == "BF") {
            config->feature_matcher = FMatch::BF;
        } else {
            if (fmatch != "FLANN") {
                std::cerr << "Warning: Unknown feature matcher \"" << fmatch << "\". Using \"FLANN\" instead." << std::endl;
            }

            config->feature_matcher = FMatch::FLANN;
        }


        config->lowe_ratio = parser.get<float>("lowe-ratio");


        config->ransac_thres = parser.get<float>("ransac-thres");


        String disparity_algorithm = parser.get<String>("disp-map");
        if (disparity_algorithm == "CustomBM") {
            config->disp_map_alg = CustomBM;
        }
        else if (disparity_algorithm == "SBM") {
            config->disp_map_alg = SBM;
        } else {
            if (disparity_algorithm != "SGBM") {
                std::cerr << "Warning: Unknown disparity map algorithm \"" << disparity_algorithm << "\". Using \"SGBM\" instead." << std::endl;
            }

            config->disp_map_alg = SGBM;
        }

#ifndef NOCONTRIB
        bool no_filter = parser.has("no-filter");
        if (!no_filter) {
            switch (config->disp_map_alg) {
                case DispMap::SBM:
                case DispMap::SGBM:
                    break;
                default:
                    std::cerr << "Warning: Can only apply filtering when using \"SBM\" or \"SGBM\" matchers." << std::endl;
                    no_filter = true;
                break;
            }
        }
        config->no_disp_filter = no_filter;
#endif

        config->focal_length = parser.get<float>("focal-length");
        config->repr_disp_min = parser.get<float>("repr-disp-min");
        config->repr_disp_max = parser.get<float>("repr-disp-max");

        config->pointcloud_only = parser.has("pointcloud-only");
        config->mesh_epsilon = parser.get<double>("mesh-epsilon");
    }
};

std::optional<Config> Config::config = std::optional<Config>();


//
// Custom block matching
//

#define GET_SET_OVR(type,name_cv,name_snk)\
    private:\
        type _##name_snk;\
    public:\
        type get##name_cv() const override {\
            return _##name_snk;\
        }\
        void set##name_cv(type name_snk) override {\
            _##name_snk = name_snk;\
        }\


class CustomBM : public StereoMatcher {
private:
    int _max_disparity = 48;
    int _window_size = 11;
public:
    GET_SET_OVR(int, MinDisparity, min_disparity);
    GET_SET_OVR(int, NumDisparities, num_disparities);
    GET_SET_OVR(int, BlockSize, block_size);
    GET_SET_OVR(int, SpeckleWindowSize, speckle_window_size);
    GET_SET_OVR(int, SpeckleRange, speckle_range);
    GET_SET_OVR(int, Disp12MaxDiff, disp_12_max_diff);

    static Ptr<CustomBM> create(int min_disparity = 0, int max_disparity = 48, int window_size = 11) {
        auto ptr = Ptr<CustomBM>(new CustomBM());
        ptr->setMinDisparity(min_disparity);
        ptr->_max_disparity = max_disparity;
        ptr->_window_size = window_size;

        return ptr;
    }

    void compute(InputArray left, InputArray right, OutputArray disparity) override {
        int disp_range = _max_disparity - _min_disparity;
        Mat kernel(_window_size, _window_size, CV_32FC1, 1.0 / _window_size);

        std::vector<Mat> disparity_maps;
        disparity_maps.reserve(disp_range);

        Mat left_mat = left.getMat();
        left_mat.convertTo(left_mat, CV_32FC1);
        Mat right_mat = right.getMat();
        right_mat.convertTo(right_mat, CV_32FC1);

        for (int d = _min_disparity; d < _max_disparity; d++) {
            float translation_data[] = { 1, 0, (float)d, 0, 1, 0};
            Mat translation_matrix(2, 3, CV_32FC1, translation_data);

            Mat shifted_right;
            warpAffine(right_mat, shifted_right, translation_matrix, right.size());

            Mat sad = abs(left_mat - shifted_right);

            Mat filtered_image(sad.size(), CV_32FC1);
            filter2D(sad, filtered_image, -1, kernel);

            disparity_maps.push_back(filtered_image);
        }

        int rows = disparity_maps[0].rows;
        int cols = disparity_maps[0].cols;
        Mat disparity_mat(rows, cols, CV_16UC1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int argmin = 0;
                float argmin_val = INFINITY;

                for (int d = 0; d < _max_disparity; d++) {
                    float val = disparity_maps[d].at<float>(i, j);

                    if (val < argmin_val) {
                        argmin_val = val;
                        argmin = d;
                    }
                }

                disparity_mat.at<uint16_t>(i, j) = (UINT16_MAX / _max_disparity) * argmin;
            }
        }

        disparity_mat.copyTo(disparity);
    }
};


//
// Mesh structure
//

struct Mesh {
    std::vector<std::tuple<Point3d, std::array<uint8_t,3>>> vertices;

    std::vector<std::array<uint32_t,3>> triangles;

    void write(std::ostream &stream) {
        const String header1 =
            "ply\n"
            "format ascii 1.0\n"
            "element vertex ";

        const String header2 =
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "element face ";

        const String header3 =
            "property list uchar int vertex_indices\n"
            "end_header";

        stream << header1 << vertices.size() << std::endl;
        stream << header2 << triangles.size() << std::endl;
        stream << header3 << std::endl;

        for (auto& vertex : vertices) {
            auto& pos = std::get<0>(vertex);
            auto& col = std::get<1>(vertex);

            stream <<  pos.x << " " <<  pos.y << " " <<  pos.z << " ";
            // We need the `+`s beacuse otherwise these values will be printed as ASCII characters
            stream << +col[0] << " " << +col[1] << " " << +col[2] << "\n";
        }

        for (auto& tri : triangles) {
            stream << 3 << " " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
        }
    }
};

//
// Debug output stuff
//

void write_img_pair(String prefix, String ext, std::array<Mat, 2>& imgs) {
    auto& [img1, img2] = imgs;

    imwrite(prefix + "__" + Config::get().img1_name_no_ext + "." + ext, img1);
    imwrite(prefix + "__" + Config::get().img2_name_no_ext + "." + ext, img2);
}

void write_comb_img(String prefix, String ext, Mat& img) {
    imwrite(prefix + "__" + Config::get().img1_name_no_ext + "__" + Config::get().img2_name_no_ext + "." + ext, img);
}


void write_feature_points(std::array<Mat, 2>& imgs, std::array<std::vector<KeyPoint>, 2>& features) {
    auto [img1, img2] = imgs;

    auto& [f1, f2] = features;

    std::array<Mat, 2> feature_imgs = { Mat(), Mat() };
    auto& [f_img1, f_img2] = feature_imgs;

    drawKeypoints(img1, f1, f_img1);
    drawKeypoints(img2, f2, f_img2);

    write_img_pair("features", "png", feature_imgs);
}

void write_feature_matches(
    std::array<Mat, 2>& imgs,
    std::array<std::vector<KeyPoint>, 2>& features,
    std::vector<DMatch>& matches)
{
    auto& [img1, img2] = imgs;
    auto& [f1, f2] = features;

    std::cerr << "Found " << matches.size() << " matches." << std::endl;

    Mat img_matches;
    drawMatches(
        img1,
        f1,
        img2,
        f2,
        matches,
        img_matches);

    write_comb_img("matches", "png", img_matches);
}

void write_rect_imgs(std::array<Mat, 2>& rect_imgs) {
    write_img_pair("rect", "png", rect_imgs);
}

void write_disp_map(Mat& disp_map_16) {
    Mat disp_map(disp_map_16.size(), CV_8UC1);
    normalize(disp_map_16, disp_map, 0, 255, NORM_MINMAX, CV_8UC1);
    write_comb_img("disp_map", "png", disp_map);
}


//
// Config -> Code stuff
//

Ptr<Feature2D> get_feature_detect_descr(Config::FDetectDescr fdetect_descr) {
    switch(fdetect_descr) {
        case Config::FDetectDescr::ORB:
            return ORB::create();
        case Config::FDetectDescr::SIFT:
#ifdef NOCONTRIB
        default:
#endif
            return SIFT::create();
#ifndef NOCONTRIB
        case Config::FDetectDescr::SURF:
        default:
            return SURF::create();
#endif
    }
}

Ptr<Feature2D> get_feature_detector() {
    return get_feature_detect_descr(Config::get().feature_detector);
}

Ptr<Feature2D> get_feature_descriptor() {
    return get_feature_detect_descr(Config::get().feature_descriptor);
}


Ptr<DescriptorMatcher> get_feature_matcher() {
    auto feature_descr = Config::get().feature_descriptor;
    switch (Config::get().feature_matcher) {
        case Config::FMatch::BF:
            int norm_type;
            switch (feature_descr) {
                case Config::FDetectDescr::ORB:
                    norm_type = NORM_HAMMING;
                    break;
                case Config::FDetectDescr::SIFT:
#ifndef NOCONTRIB
                case Config::FDetectDescr::SURF:
#endif
                default:
                    norm_type = NORM_L2;
                    break;
            }
            return BFMatcher::create(norm_type);
        default:
            Ptr<flann::IndexParams> index_params;
            switch (feature_descr) {
                case Config::FDetectDescr::ORB:
                    index_params = Ptr<flann::IndexParams>(new flann::LshIndexParams(12, 20, 2));
                    break;
                case Config::FDetectDescr::SIFT:
#ifndef NOCONTRIB
                case Config::FDetectDescr::SURF:
#endif
                default:
                    index_params = Ptr<flann::IndexParams>(new flann::KDTreeIndexParams(4));
                    break;
            }
            auto matcher = Ptr<DescriptorMatcher>(new FlannBasedMatcher(index_params));
            return matcher;
    }
}


Ptr<StereoMatcher> get_stereo_matcher() {
    switch(Config::get().disp_map_alg) {
        case Config::DispMap::CustomBM:
            return CustomBM::create();
        case Config::DispMap::SBM:
            return StereoBM::create();
        default:
            return StereoSGBM::create(0, 64, 7);
    }
}


struct RectifyResults {
    Mat R1;
    Mat R2;
    Mat P1;
    Mat P2;
    Mat Q;
};

struct ReprojectionResult {
    Mat points_3d;
    Mat color;
    Mat mask;
    int valid_points;

    std::tuple<Mesh,Mat> to_mesh() {
        Mesh mesh;
        Mat index_mat(points_3d.size(), CV_32S, -1);

        for (int i = 0; i < points_3d.rows; i++) {
            const Vec3f* image_3d_ptr = points_3d.ptr<Vec3f>(i);
            for (int j = 0; j < points_3d.cols; j++) {
                if (mask.at<uint8_t>(i,j) == 0) continue;
                
                auto& pos = image_3d_ptr[j];
                auto col = color.ptr<Point3_<uint8_t>>(i, j);

                mesh.vertices.push_back(
                    std::make_tuple(
                        Point3d { pos[0], pos[1], pos[2] },
                        std::array<uint8_t,3> { col->z, col->y, col->x }
                    ));

                index_mat.at<int32_t>(i,j) = mesh.vertices.size() - 1;
            }
        }

        return std::make_tuple(mesh, index_mat);
    }
};

class Pipeline {
private:
    std::optional<std::array<Mat, 2>> imgs;
    std::optional<std::array<Mat, 2>> gs_imgs;
    std::optional<std::array<std::vector<KeyPoint>, 2>> features;
    std::optional<std::array<Mat, 2>> feature_descriptors;
    std::optional<std::tuple<std::vector<DMatch>, std::array<std::vector<Point2d>, 2>>> feature_matches;
    std::optional<Mat> fundamental_mat;
    std::optional<std::tuple<Mat,Mat>> essential_params;
    std::optional<std::tuple<Mat,Mat>> pose;
    std::optional<RectifyResults> rectify_results;
    std::optional<std::array<Mat, 2>> homographies;
    std::optional<std::array<Mat, 2>> rect_imgs;
    std::optional<std::array<Mat, 2>> gs_rect_imgs;
#ifndef NOCONTRIB
    std::optional<std::array<Ptr<StereoMatcher>, 2>> stereo_matchers;
    std::optional<std::array<Mat, 2>> unfiltered_disparity_maps;
    std::optional<Mat> filtered_disparity_map;
#endif
    std::optional<Mat> disparity_map;
    std::optional<ReprojectionResult> reprojection;
    std::optional<Mesh> mesh;

public:
    std::array<Mat, 2>& get_imgs() {
        if (!imgs) {
            imgs = { imread(Config::get().img1_name, IMREAD_COLOR),
                     imread(Config::get().img2_name, IMREAD_COLOR) };
        }

        return *imgs;
    }

    std::array<Mat, 2>& get_gs_imgs() {
        if (!gs_imgs) {
            auto& [img1, img2] = get_imgs();

            gs_imgs = { Mat(img1.rows, img1.cols, CV_8UC1), Mat(img2.rows, img2.cols, CV_8UC1) };
            auto& [gs_img1, gs_img2] = *gs_imgs;

            cvtColor(img1, gs_img1, COLOR_BGR2GRAY);
            cvtColor(img2, gs_img2, COLOR_BGR2GRAY);
        }

        return *gs_imgs;
    }

    std::array<std::vector<KeyPoint>, 2>& get_features() {
        if (!features) {
            auto feature_detector = get_feature_detector();

            auto& [gs_img1, gs_img2] = get_gs_imgs();
            features = { std::vector<KeyPoint>(), std::vector<KeyPoint>() };
            auto& [f1, f2] = *features;

            feature_detector->detect(gs_img1, f1);
            feature_detector->detect(gs_img2, f2);

            if (Config::get().analysis) write_feature_points(get_imgs(), get_features());
        }

        return *features;
    }

    std::array<Mat, 2>& get_feature_descriptors() {
        if (!feature_descriptors) {
            auto feature_descriptor = get_feature_descriptor();

            auto& [gs_img1, gs_img2] = get_gs_imgs();
            auto& [f1, f2] = get_features();

            feature_descriptors = { Mat(), Mat() };
            auto& [fd1, fd2] = *feature_descriptors;

            feature_descriptor->compute(gs_img1, f1, fd1);
            feature_descriptor->compute(gs_img2, f2, fd2);
        }

        return *feature_descriptors;
    }

    std::tuple<std::vector<DMatch>, std::array<std::vector<Point2d>, 2>>& get_feature_matches() {
        if (!feature_matches) {
            auto feature_matcher = get_feature_matcher();

            auto& [f1, f2] = get_features();
            auto& [fd1, fd2] = get_feature_descriptors();

            feature_matches = { std::vector<DMatch>(), std::array<std::vector<Point2d>, 2>() };
            auto& [matches, matching_features] = *feature_matches;
            auto& [mf1, mf2] = matching_features;

            std::vector<std::vector<DMatch>> knn_matches;
            feature_matcher->knnMatch(fd1, fd2, knn_matches, 2);

            std::sort(
                knn_matches.begin(),
                knn_matches.end(),
                [](std::vector<DMatch> x, std::vector<DMatch> y) {
                    if (x.size() == 0) return false;
                    if (y.size() == 0) return true;

                    return x[0].distance < y[0].distance;
                });

            size_t relevant_matches = 1024;
            for (auto& m: knn_matches) {
                if (relevant_matches == 0) break;

                if (m[0].distance / m[1].distance < Config::get().lowe_ratio) {
                    matches.push_back(m[0]);
                    mf1.push_back(f1[m[0].queryIdx].pt);
                    mf2.push_back(f2[m[0].trainIdx].pt);

                    relevant_matches--;
                }
            }

            if (Config::get().analysis) write_feature_matches(get_imgs(), get_features(), matches);
        }

        return *feature_matches;
    }

    Mat get_camera_matrix() {
        auto& [img, _] = get_imgs();

        Mat camera = Mat::eye(3, 3, CV_32F);
        camera.at<float>(0,0) = Config::get().focal_length;
        camera.at<float>(1,1) = Config::get().focal_length;
        camera.at<float>(0,2) = img.cols / 2.0;
        camera.at<float>(1,2) = img.rows / 2.0;

        return camera;
    }

    std::tuple<Mat,Mat>& get_essential_params() {
        if (!essential_params) {
            auto& [_, fs] = get_feature_matches();
            Mat mask;
            auto essential_mat = findEssentialMat(fs[0], fs[1], get_camera_matrix(), RANSAC, 0.99, 1.0, mask);

            essential_params = std::make_tuple<Mat,Mat>(std::move(essential_mat), std::move(mask));

            if (Config::get().analysis) {
                std::cerr << "Essential matrix: " << std::get<0>(*essential_params) << std::endl;
            }
        }

        return *essential_params;
    }

    std::tuple<Mat,Mat>& get_pose() {
        if (!pose) {
            auto& [essential_mat, mask] = get_essential_params();
            auto& [_, fs] = get_feature_matches();

            Mat rotation, translation;
            recoverPose(essential_mat, fs[0], fs[1], get_camera_matrix(), rotation, translation, mask);

            pose = std::make_tuple<Mat,Mat>(std::move(rotation), std::move(translation));

            if (Config::get().analysis) {
                std::cerr << "Rotation: "    << std::get<0>(*pose) << std::endl;
                std::cerr << "Translation: " << std::get<1>(*pose) << std::endl;
            }
        }

        return *pose;
    }

    RectifyResults& get_rectify_results() {
        if (!rectify_results) {
            auto& [img1, _] = get_imgs();
            auto& [rotation, translation] = get_pose();

            Mat R1, R2, P1, P2, Q;

            stereoRectify(
                get_camera_matrix(),
                Mat::zeros(1, 4, CV_32F),
                get_camera_matrix(),
                Mat::zeros(1, 4, CV_32F),
                img1.size(),
                rotation,
                translation,
                R1,
                R2,
                P1,
                P2,
                Q
            );

            rectify_results = {
                R1,
                R2,
                P1,
                P2,
                Q
            };

            if (Config::get().analysis) {
                std::cerr << "R1: " << rectify_results->R1 << std::endl;
                std::cerr << "R2: " << rectify_results->R2 << std::endl;
                std::cerr << "P1: " << rectify_results->P1 << std::endl;
                std::cerr << "P2: " << rectify_results->P2 << std::endl;
                std::cerr << "Q: " << rectify_results->Q << std::endl;
            }
        }

        return *rectify_results;
    }

    std::array<Mat, 2>& get_rect_imgs() {
        if (!rect_imgs) {
            auto [img1, img2] = get_imgs();
            auto& rectify_results = get_rectify_results();

            Mat map1x, map1y, map2x, map2y;
            initUndistortRectifyMap(
                get_camera_matrix(),
                Mat::zeros(1, 4, CV_32F),
                rectify_results.R1,
                rectify_results.P1,
                img1.size(),
                CV_32FC1,
                map1x,
                map1y
            );
            initUndistortRectifyMap(
                get_camera_matrix(),
                Mat::zeros(1, 4, CV_32F),
                rectify_results.R2,
                rectify_results.P1,
                img2.size(),
                CV_32FC1,
                map2x,
                map2y
            );


            rect_imgs = { Mat(img1.size(), img1.type()), Mat(img2.size(), img2.type()) };
            auto& [rimg1, rimg2] = *rect_imgs;

            remap(img1, rimg1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT);
            remap(img2, rimg2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT);

            if (Config::get().analysis) write_rect_imgs(get_rect_imgs());
        }

        return *rect_imgs;
    }

    std::array<Mat, 2>& get_gs_rect_imgs() {
        if (!gs_rect_imgs) {
            auto& [rimg1, rimg2] = get_rect_imgs();

            gs_rect_imgs = { Mat(rimg1.size(), CV_8UC1), Mat(rimg2.size(), CV_8UC1) };
            auto& [gs_rimg1, gs_rimg2] = *gs_rect_imgs;

            cvtColor(rimg1, gs_rimg1, COLOR_BGR2GRAY);
            cvtColor(rimg2, gs_rimg2, COLOR_BGR2GRAY);
        }

        return *gs_rect_imgs;
    }

    std::array<Mat, 2>& get_disp_src_imgs() {
        switch (Config::get().disp_map_alg) {
            case Config::DispMap::CustomBM:
            case Config::DispMap::SBM:
                return Config::get().no_rect_imgs ? get_gs_imgs() : get_gs_rect_imgs();
            default:
                return Config::get().no_rect_imgs ? get_imgs() : get_rect_imgs();
        }
    }

#ifndef NOCONTRIB
    std::array<Ptr<StereoMatcher>, 2>& get_stereo_matchers() {
        if (!stereo_matchers) {
            auto left_matcher = get_stereo_matcher();

            Ptr<StereoMatcher> right_matcher;
            switch(Config::get().disp_map_alg) {
                case Config::DispMap::SBM:
                case Config::DispMap::SGBM:
                    right_matcher = createRightMatcher(left_matcher);
                    break;
                default:
                    right_matcher = get_stereo_matcher();
                    break;
            }

            stereo_matchers = { left_matcher, right_matcher };
        }

        return *stereo_matchers;
    }

    std::array<Mat, 2>& get_unfiltered_disparity_maps() {
        if (!unfiltered_disparity_maps) {
            auto& [dimg1, dimg2] = get_disp_src_imgs();
            auto& [sm1, sm2] = get_stereo_matchers();

            unfiltered_disparity_maps = { Mat(dimg1.size(), CV_16SC1), Mat(dimg2.size(), CV_16SC1) };
            auto& [dmap1, dmap2] = *unfiltered_disparity_maps;

            sm1->compute(dimg1, dimg2, dmap1);
            sm2->compute(dimg2, dimg1, dmap2);
        }

        return *unfiltered_disparity_maps;
    }

    Mat& get_filtered_disparity_map() {
        if (!filtered_disparity_map) {
            auto& [sm1, _sm2] = get_stereo_matchers();
            auto filter = createDisparityWLSFilter(sm1);

            auto& [dmap1, dmap2] = get_unfiltered_disparity_maps();
            auto& [dimg1, _dimg2] = get_disp_src_imgs();

            filtered_disparity_map = Mat(dimg1.size(), CV_16SC1);

            filter->setLambda(8000.0);
            filter->setSigmaColor(1.5);
            filter->filter(dmap1, dimg1, *filtered_disparity_map, dmap2);
        }

        return *filtered_disparity_map;
    }

    Mat& get_disparity_map() {
        if (!disparity_map) {
            disparity_map = 
                Config::get().no_disp_filter ?
                    std::get<0>(get_unfiltered_disparity_maps()) :
                    get_filtered_disparity_map();

            if (Config::get().analysis) write_disp_map(get_disparity_map());
        }

        return *disparity_map;
    }
#else
    Mat& get_disparity_map() {
        if (!disparity_map) {
            auto& [dimg1, dimg2] = get_disp_src_imgs();

            auto sm = get_stereo_matcher();

            disparity_map = Mat(dimg1.size(), CV_16SC1);
            sm->compute(dimg1, dimg2, *disparity_map);

            if (Config::get().analysis) write_disp_map(get_disparity_map());
        }

        return *disparity_map;
    }
#endif

    ReprojectionResult& get_reprojection() {
        if (!reprojection) {
            auto [img1, _] = get_rect_imgs();
            auto rectify_results = get_rectify_results();
            auto disparity_map = get_disparity_map();

            disparity_map.convertTo(disparity_map, CV_32F, 1.0 / 16.0);
            
            int valid_points = 0;
            Mat mask = Mat(disparity_map.size(), CV_8UC1);
            for (int i = 0; i < disparity_map.rows; i++) {
                for (int j = 0; j < disparity_map.cols; j++) {
                    float disp = disparity_map.at<float>(i,j);

                    uint8_t mask_val = 1;
                    if (disp <= Config::get().repr_disp_min || disp >= Config::get().repr_disp_max) mask_val = 0;

                    valid_points += mask_val;
                    mask.at<uint8_t>(i,j) = mask_val;
                }
            }

            Mat points_3d;
            reprojectImageTo3D(disparity_map, points_3d, rectify_results.Q);
            
            reprojection = {
                points_3d,
                img1,
                mask,
                valid_points
            };
        }

        return *reprojection;
    }

    Mesh& get_mesh() {
        if (!mesh) {
            auto& repr = get_reprojection();
            auto [the_mesh, indices] = repr.to_mesh();

            double epsilon = Config::get().mesh_epsilon;

            if (!Config::get().pointcloud_only) {
                for (int i = 0; i < indices.rows - 1; i++) {
                    for (int j = 0; j < indices.cols - 1; j++) {
                        unsigned int ix0 = indices.at<int32_t>(i,j);
                        if (ix0 == -1) continue;
                        
                        unsigned int ix1 = indices.at<int32_t>(i+1,j+1);
                        if (ix1 == -1) continue;

                        Vec3d p0 = std::get<0>(the_mesh.vertices[ix0]);
                        Vec3d p1 = std::get<0>(the_mesh.vertices[ix1]);

                        if (norm(p1 - p0) > epsilon) continue;

                        unsigned int ixA = indices.at<int32_t>(i+1,j);
                        if (ixA != -1) {
                            Vec3d pA = std::get<0>(the_mesh.vertices[ixA]);

                            if (norm(p0 - pA) < epsilon && norm(p1 - pA) < epsilon) {
                                the_mesh.triangles.push_back({ ix0, ix1, ixA });
                            }
                        }

                        unsigned int ixB = indices.at<int32_t>(i,j+1);
                        if (ixB != -1) {
                            Vec3d pB = std::get<0>(the_mesh.vertices[ixB]);

                            if (norm(p0 - pB) < epsilon && norm(p1 - pB) < epsilon) {
                                the_mesh.triangles.push_back({ ix0, ixB, ix1 });
                            }
                        }
                    }
                }
            }
            
            mesh = the_mesh;
        }

        return *mesh;
    }
};

int main(int argc, char** argv)
{
    Config::init(argc, argv);

    Pipeline pipeline;
    auto mesh = pipeline.get_mesh();

    std::ofstream mesh_file("mesh__" + Config::get().img1_name_no_ext + "." + "ply");
    mesh.write(mesh_file);
    mesh_file.close();

    return 0;
}
