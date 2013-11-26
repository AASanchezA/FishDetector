//Timer Class used to measure process time
#ifndef TIMER_HPP
#define TIMER_HPP

#include <opencv2/core/core.hpp>
#include <cstdio>
#include <iostream>

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
	Timer() : start_(0), time_(0) {}

	void start()
	{
		start_ = cv::getTickCount();
	}

	void stop()
	{
		CV_Assert(start_ != 0);
		int64 end = cv::getTickCount();
		time_ += end - start_;
		start_ = 0;
	}

	double time()
	{
		double ret = time_ / cv::getTickFrequency();
		time_ = 0;
		return ret;
	}

private:
	int64 start_, time_;
};
#endif
