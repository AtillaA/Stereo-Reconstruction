#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <opencv2/core/types.hpp>
#include <string>


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZRGBNormal PointTN;

int main(int argc, char** argv)
{

	//Load input file
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_downSampled(new pcl::PointCloud<PointT>);
	clock_t start, end;

	start = clock();
	std::cout <<"START READING" << std::endl;
	if (pcl::io::loadPLYFile("cones1.ply", *cloud) == -1)
	{
		std::cout << "Failed to read the point cloud£¡" << std::endl;
	}
	std::cout << "Orginal points number: " << cloud->points.size() << std::endl;
	// down sampling
	pcl::VoxelGrid<PointT> downSampled;  //create filter
	downSampled.setInputCloud(cloud);            //set input point cloud
	downSampled.setLeafSize(0.1f, 0.1f, 0.1f);  //1cm cube into 1 point
	downSampled.filter(*cloud_downSampled);
	std::cout << "down-sampling points number: " << cloud_downSampled->points.size() << std::endl;
	// remove outlier
	pcl::StatisticalOutlierRemoval<PointT> sor;   //create filter
	sor.setInputCloud(cloud_downSampled);            //set input point cloud
	sor.setMeanK(50);                               //set number of points considered
	sor.setStddevMulThresh(1.0);                      //threshold of whether it's a outlier
	sor.filter(*cloud);                  //save output
	std::cout << "remove-outlier points number: " << cloud->points.size() << std::endl;

	// normal estimation   
	pcl::NormalEstimation<PointT, pcl::Normal> normalEstimation;                    //output
	normalEstimation.setInputCloud(cloud);                                    //input cloud
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	normalEstimation.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// ksearch or radius
	normalEstimation.setKSearch(10);                    // nearest 10 points
	//normalEstimation.setRadiusSearch(0.03);            //radius
	normalEstimation.compute(*normals);
	// combination
	pcl::PointCloud<PointTN>::Ptr cloud_with_normals(new pcl::PointCloud<PointTN>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	//pcl::io::savePLYFile("conenormal.ply", *cloud_with_normals);
	
	// greed projection triangulation 
	pcl::search::KdTree<PointTN>::Ptr tree2(new pcl::search::KdTree<PointTN>);
	tree2->setInputCloud(cloud_with_normals);
	pcl::GreedyProjectionTriangulation<PointTN> gp3;
	pcl::PolygonMesh triangles; //output
	gp3.setSearchRadius(20000);  //search radius
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);

	//gp3.setMinimumAngle(M_PI / 18); // min 10¡ã
	//gp3.setMaximumAngle(2 * M_PI / 3); // max 120¡ã

	//gp3.setMaximumSurfaceAngle(M_PI / 4); // max 45¡ã for normals
	gp3.setNormalConsistency(false);

	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);
	pcl::io::savePLYFile("coneoutrgb2.ply", triangles);
	end = clock();
	std::cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << std::endl;
	return 1;
}
