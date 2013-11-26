/*
 * functions.hpp
 *
 *  Created on: Nov 26, 2013
 *      Author: andres
 */

#ifndef FUNCTIONS_HPP_
#define FUNCTIONS_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>
#include <vector>
#include "ConfigFile.h"

enum sendTo
{
	NONE  = 0,
	RIGHT = 1,
	LEFT  = 2
};


const double cv_pi = 3.141592653589;

// Function prototypes
void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f);

void subtractPlaneSint(const cv::Mat& color, cv::Mat& mask);

std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
		int num_modalities, cv::Point offset, cv::Size size,
		cv::Mat& mask, cv::Mat& dst);

void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
		int num_modalities, cv::Point offset, cv::Size size,
		cv::Mat& dst);

void drawResponse(const std::vector<cv::linemod::Template>& templates,
		int num_modalities, cv::Mat& dst, cv::Point offset, int T);

void drawBoxes(std::vector<cv::Rect>& boxes, cv::Mat& dst, const float factor=1);

//generate a canvas of three images
cv::Mat display3(std::vector<cv::Mat>& images);

//Check if template from Head and Tail are connected
bool isFish(cv::linemod::Match& mHead, cv::linemod::Match& mTail, int num_modalities, float threshold, cv::Mat& dst,
		cv::Ptr<cv::linemod::Detector>& detHead,
		cv::Ptr<cv::linemod::Detector>& detTail,
		ConfigFile &config);

cv::Mat displayQuantized(const cv::Mat& quantized);

//Calculate spring equation with acceleration equal 0
double springEq(cv::Point pt_1, cv::Point pt_2, cv::Vec2d& speed, double& teta);

//Prepare mask for training linemod
void subtractMask( cv::vector<cv::Mat>& color, cv::Mat& mask, cv::vector<cv::Mat>& masked, int& maskside);

//Find orientation of templates
int findTemplateSide(cv::vector<cv::Point> contour, cv::RotatedRect elipse, cv::Rect rect, cv::Mat& dst, cv::vector<cv::Point>& extrem);

//Sort contour point clockwise
cv::vector<cv::Point> sortExtremContour(cv::vector<cv::Point>& contour, cv::RotatedRect& elipse, cv::Rect& rect, cv::Mat& dst);

//define if a point in contour is more to left of center
bool less(cv::Point a, cv::Point b, cv::Point center);


#endif /* FUNCTIONS_HPP_ */
