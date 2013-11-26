/*
 * functions.cpp
 *
 *  Created on: Nov 26, 2013
 *      Author: andres
 */

#include "functions.hpp"


static void reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f)
{
	real.resize(proj.size());
	double f_inv = 1.0 / f;

	for (int i = 0; i < (int)proj.size(); ++i)
	{
		double Z  = proj[i].z;
		real[i].x = (proj[i].x - 320.) * (f_inv * Z);
		real[i].y = (proj[i].y - 240.) * (f_inv * Z);
		real[i].z = Z;
	}
}


static void filterPlane(IplImage * ap_depth, std::vector<IplImage *> & a_masks, std::vector<CvPoint> & a_chain, double f)
{
  const int l_num_cost_pts = 200;

  float l_thres = 4;

  IplImage * lp_mask = cvCreateImage(cvGetSize(ap_depth), IPL_DEPTH_8U, 1);
  cvSet(lp_mask, cvRealScalar(0));

  std::vector<CvPoint> l_chain_vector;

  float l_chain_length = 0;
  float * lp_seg_length = new float[a_chain.size()];

  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    float x_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x);
    float y_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y);
    lp_seg_length[l_i] = sqrt(x_diff*x_diff + y_diff*y_diff);
    l_chain_length += lp_seg_length[l_i];
  }
  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    if (lp_seg_length[l_i] > 0)
    {
      int l_cur_num = cvRound(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length);
      float l_cur_len = lp_seg_length[l_i] / l_cur_num;

      for (int l_j = 0; l_j < l_cur_num; ++l_j)
      {
        float l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]);

        CvPoint l_pts;

        l_pts.x = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x) + a_chain[l_i].x);
        l_pts.y = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y) + a_chain[l_i].y);

        l_chain_vector.push_back(l_pts);
      }
    }
  }
  std::vector<cv::Point3d> lp_src_3Dpts(l_chain_vector.size());

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    lp_src_3Dpts[l_i].x = l_chain_vector[l_i].x;
    lp_src_3Dpts[l_i].y = l_chain_vector[l_i].y;
    lp_src_3Dpts[l_i].z = CV_IMAGE_ELEM(ap_depth, unsigned short, cvRound(lp_src_3Dpts[l_i].y), cvRound(lp_src_3Dpts[l_i].x));
    //CV_IMAGE_ELEM(lp_mask,unsigned char,(int)lp_src_3Dpts[l_i].Y,(int)lp_src_3Dpts[l_i].X)=255;
  }
  //cv_show_image(lp_mask,"hallo2");

  reprojectPoints(lp_src_3Dpts, lp_src_3Dpts, f);

  CvMat * lp_pts = cvCreateMat((int)l_chain_vector.size(), 4, CV_32F);
  CvMat * lp_v = cvCreateMat(4, 4, CV_32F);
  CvMat * lp_w = cvCreateMat(4, 1, CV_32F);

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    CV_MAT_ELEM(*lp_pts, float, l_i, 0) = (float)lp_src_3Dpts[l_i].x;
    CV_MAT_ELEM(*lp_pts, float, l_i, 1) = (float)lp_src_3Dpts[l_i].y;
    CV_MAT_ELEM(*lp_pts, float, l_i, 2) = (float)lp_src_3Dpts[l_i].z;
    CV_MAT_ELEM(*lp_pts, float, l_i, 3) = 1.0f;
  }
  cvSVD(lp_pts, lp_w, 0, lp_v);

  float l_n[4] = {CV_MAT_ELEM(*lp_v, float, 0, 3),
                  CV_MAT_ELEM(*lp_v, float, 1, 3),
                  CV_MAT_ELEM(*lp_v, float, 2, 3),
                  CV_MAT_ELEM(*lp_v, float, 3, 3)};

  float l_norm = sqrt(l_n[0] * l_n[0] + l_n[1] * l_n[1] + l_n[2] * l_n[2]);

  l_n[0] /= l_norm;
  l_n[1] /= l_norm;
  l_n[2] /= l_norm;
  l_n[3] /= l_norm;

  float l_max_dist = 0;

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    float l_dist =  l_n[0] * CV_MAT_ELEM(*lp_pts, float, l_i, 0) +
                    l_n[1] * CV_MAT_ELEM(*lp_pts, float, l_i, 1) +
                    l_n[2] * CV_MAT_ELEM(*lp_pts, float, l_i, 2) +
                    l_n[3] * CV_MAT_ELEM(*lp_pts, float, l_i, 3);

    if (fabs(l_dist) > l_max_dist)
      l_max_dist = l_dist;
  }
  //std::cerr << "plane: " << l_n[0] << ";" << l_n[1] << ";" << l_n[2] << ";" << l_n[3] << " maxdist: " << l_max_dist << " end" << std::endl;
  int l_minx = ap_depth->width;
  int l_miny = ap_depth->height;
  int l_maxx = 0;
  int l_maxy = 0;

  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    l_minx = std::min(l_minx, a_chain[l_i].x);
    l_miny = std::min(l_miny, a_chain[l_i].y);
    l_maxx = std::max(l_maxx, a_chain[l_i].x);
    l_maxy = std::max(l_maxy, a_chain[l_i].y);
  }
  int l_w = l_maxx - l_minx + 1;
  int l_h = l_maxy - l_miny + 1;
  int l_nn = (int)a_chain.size();

  CvPoint * lp_chain = new CvPoint[l_nn];

  for (int l_i = 0; l_i < l_nn; ++l_i)
    lp_chain[l_i] = a_chain[l_i];

  cvFillPoly(lp_mask, &lp_chain, &l_nn, 1, cvScalar(255, 255, 255));

  delete[] lp_chain;

  //cv_show_image(lp_mask,"hallo1");

  std::vector<cv::Point3d> lp_dst_3Dpts(l_h * l_w);

  int l_ind = 0;

  for (int l_r = 0; l_r < l_h; ++l_r)
  {
    for (int l_c = 0; l_c < l_w; ++l_c)
    {
      lp_dst_3Dpts[l_ind].x = l_c + l_minx;
      lp_dst_3Dpts[l_ind].y = l_r + l_miny;
      lp_dst_3Dpts[l_ind].z = CV_IMAGE_ELEM(ap_depth, unsigned short, l_r + l_miny, l_c + l_minx);
      ++l_ind;
    }
  }
  reprojectPoints(lp_dst_3Dpts, lp_dst_3Dpts, f);

  l_ind = 0;

  for (int l_r = 0; l_r < l_h; ++l_r)
  {
    for (int l_c = 0; l_c < l_w; ++l_c)
    {
      float l_dist = (float)(l_n[0] * lp_dst_3Dpts[l_ind].x + l_n[1] * lp_dst_3Dpts[l_ind].y + lp_dst_3Dpts[l_ind].z * l_n[2] + l_n[3]);

      ++l_ind;

      if (CV_IMAGE_ELEM(lp_mask, unsigned char, l_r + l_miny, l_c + l_minx) != 0)
      {
        if (fabs(l_dist) < std::max(l_thres, (l_max_dist * 2.0f)))
        {
          for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
          {
            int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
            int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

            CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 0;
          }
        }
        else
        {
          for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
          {
            int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
            int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

            CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 255;
          }
        }
      }
    }
  }
  cvReleaseImage(&lp_mask);
  cvReleaseMat(&lp_pts);
  cvReleaseMat(&lp_w);
  cvReleaseMat(&lp_v);
}


void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f)
{
	mask = cv::Mat::zeros(depth.size(), CV_8U);
	std::vector<IplImage*> tmp;
	IplImage mask_ipl = mask;
	tmp.push_back(&mask_ipl);
	IplImage depth_ipl = depth;
	filterPlane(&depth_ipl, tmp, chain, f);
}


void subtractPlaneSint(const cv::Mat& color, cv::Mat& mask)
{
	//  cv::Mat bg(color.size(), CV_8UC3, cv::Scalar(64, 64, 64));
	cv::Mat masktmp = cv::Mat::zeros(color.size(), CV_8U);
	//  mask = cv::Mat::ones(color.size(), CV_8U)*255;
	cv::copyMakeBorder(masktmp, masktmp, 1, 1, 1, 1, cv::BORDER_REPLICATE);

	std::vector<cv::Mat> diffChannels;
	cv::split(color, diffChannels);

	cv::threshold(diffChannels[2], masktmp, 230, 255, CV_THRESH_BINARY);
	mask = masktmp;
	std::cout << "size mask = " << mask.cols << ", " << mask.rows << std::endl;
	cv::namedWindow("diff", 1);
	cv::imshow("diff", masktmp);

}


std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
                                      int num_modalities, cv::Point offset, cv::Size size,
                                      cv::Mat& mask, cv::Mat& dst)
{
  templateConvexHull(templates, num_modalities, offset, size, mask);

  const int OFFSET = 30;
  cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), OFFSET);
  CvMemStorage * lp_storage = cvCreateMemStorage(0);
  CvTreeNodeIterator l_iterator;
  CvSeqReader l_reader;
  CvSeq * lp_contour = 0;

  cv::Mat mask_copy = mask.clone();
  IplImage mask_copy_ipl = mask_copy;
  cvFindContours(&mask_copy_ipl, lp_storage, &lp_contour, sizeof(CvContour),
                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  cv::namedWindow("grabcut");
  cv::imshow("grabcut", mask_copy);
  std::vector<CvPoint> l_pts1; // to use as input to cv_primesensor::filter_plane

  cvInitTreeNodeIterator(&l_iterator, lp_contour, 1);
  while ((lp_contour = (CvSeq *)cvNextTreeNode(&l_iterator)) != 0)
  {
    CvPoint l_pt0;
    cvStartReadSeq(lp_contour, &l_reader, 0);
    CV_READ_SEQ_ELEM(l_pt0, l_reader);
    l_pts1.push_back(l_pt0);

    for (int i = 0; i < lp_contour->total; ++i)
    {
      CvPoint l_pt1;
      CV_READ_SEQ_ELEM(l_pt1, l_reader);
      /// @todo Really need dst at all? Can just as well do this outside
      cv::line(dst, l_pt0, l_pt1, CV_RGB(0, 255, 0), 2);

      l_pt0 = l_pt1;
      l_pts1.push_back(l_pt0);
    }
  }
  cvReleaseMemStorage(&lp_storage);

  return l_pts1;
}


std::vector<CvPoint> x(const std::vector<cv::linemod::Template>& templates,
                                      int num_modalities, cv::Point offset, cv::Size size,
                                      cv::Mat& mask, cv::Mat& dst)
{
  templateConvexHull(templates, num_modalities, offset, size, mask);

  const int OFFSET = 30;
  cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), OFFSET);

  CvMemStorage * lp_storage = cvCreateMemStorage(0);
  CvTreeNodeIterator l_iterator;
  CvSeqReader l_reader;
  CvSeq * lp_contour = 0;

  cv::Mat mask_copy = mask.clone();
  IplImage mask_copy_ipl = mask_copy;
  cvFindContours(&mask_copy_ipl, lp_storage, &lp_contour, sizeof(CvContour),
                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  std::vector<CvPoint> l_pts1; // to use as input to cv_primesensor::filter_plane

  cvInitTreeNodeIterator(&l_iterator, lp_contour, 1);
  while ((lp_contour = (CvSeq *)cvNextTreeNode(&l_iterator)) != 0)
  {
    CvPoint l_pt0;
    cvStartReadSeq(lp_contour, &l_reader, 0);
    CV_READ_SEQ_ELEM(l_pt0, l_reader);
    l_pts1.push_back(l_pt0);

    for (int i = 0; i < lp_contour->total; ++i)
    {
      CvPoint l_pt1;
      CV_READ_SEQ_ELEM(l_pt1, l_reader);
      /// @todo Really need dst at all? Can just as well do this outside
      cv::line(dst, l_pt0, l_pt1, CV_RGB(0, 255, 0), 2);

      l_pt0 = l_pt1;
      l_pts1.push_back(l_pt0);
    }
  }
  cvReleaseMemStorage(&lp_storage);

  return l_pts1;
}


// Adapted from cv_show_angles
cv::Mat displayQuantized(const cv::Mat& quantized)
{
  cv::Mat color(quantized.size(), CV_8UC3);
  for (int r = 0; r < quantized.rows; ++r)
  {
    const uchar* quant_r = quantized.ptr(r);
    cv::Vec3b* color_r = color.ptr<cv::Vec3b>(r);

    for (int c = 0; c < quantized.cols; ++c)
    {
      cv::Vec3b& bgr = color_r[c];
      switch (quant_r[c])
      {
        case 0:   bgr[0]=  0; bgr[1]=  0; bgr[2]=  0;    break;
        case 1:   bgr[0]= 55; bgr[1]= 55; bgr[2]= 55;    break;
        case 2:   bgr[0]= 80; bgr[1]= 80; bgr[2]= 80;    break;
        case 4:   bgr[0]=105; bgr[1]=105; bgr[2]=105;    break;
        case 8:   bgr[0]=130; bgr[1]=130; bgr[2]=130;    break;
        case 16:  bgr[0]=155; bgr[1]=155; bgr[2]=155;    break;
        case 32:  bgr[0]=180; bgr[1]=180; bgr[2]=180;    break;
        case 64:  bgr[0]=205; bgr[1]=205; bgr[2]=205;    break;
        case 128: bgr[0]=230; bgr[1]=230; bgr[2]=230;    break;
        case 255: bgr[0]=  0; bgr[1]=  0; bgr[2]=255;    break;
        default:  bgr[0]=  0; bgr[1]=255; bgr[2]=  0;    break;
      }
    }
  }

  return color;
}


// Adapted from cv_line_template::convex_hull
void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
                        int num_modalities, cv::Point offset, cv::Size size,
                        cv::Mat& dst)
{
	std::vector<cv::Point> points;
	for (int m = 0; m < num_modalities; ++m)
	{
		for (int i = 0; i < (int)templates[m].features.size(); ++i)
		{
			cv::linemod::Feature f = templates[m].features[i];
			points.push_back(cv::Point(f.x, f.y) + offset);
		}
	}

	std::vector<cv::Point> hull;
	cv::convexHull(points, hull);

	dst = cv::Mat::zeros(size, CV_8U);
	const int hull_count = (int)hull.size();
	const cv::Point* hull_pts = &hull[0];
	cv::fillPoly(dst, &hull_pts, &hull_count, 1, cv::Scalar(255));
//  cv::namedWindow("grabcut");
//  cv::imshow("grabcut", dst);
}


void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
	cv::Mat temp1, temp2;
	temp1 = dst.clone();

	static int index = 0;
	static const cv::Scalar COLORS[8] = { CV_RGB(0  , 0  , 255),
	                                      CV_RGB(0  , 255, 0  ),
	                                      CV_RGB(255, 255, 0  ),
	                                      CV_RGB(255, 140, 0  ),
	                                      CV_RGB(255, 0  , 0  ),
	                                      CV_RGB(255, 0  , 255),
	                                      CV_RGB(0  , 0  , 0  ),
	                                      CV_RGB(200, 100, 255)
	                                    };

	// Countors and rotatedRect to save fit ellipse for template
	cv::RotatedRect minEllipse;
	cv::Rect minRectable;
	cv::vector<cv::Point> contour, contourin, contourout;
	cv::Point ptin, ptout;
//	cv::Scalar colorEllipse = COLORS[1];

	for (int m = 0; m < num_modalities; ++m)
	{
		for (int i = 0; i < (int)templates[m].features.size(); ++i)
		{
			index = index > 4 ? (index % 5) : index;
			cv::linemod::Feature f = templates[m].features[i];
			cv::Point pt(f.x + offset.x, f.y + offset.y);
			//      cv::circle(dst, pt, T / 2, color);
			cv::circle(dst, pt, T / 2, COLORS[index]);
			contour.push_back(pt);
		}
	}
	//find minimum ellipse and boundingbox for template draw it
	minEllipse = cv::fitEllipse(cv::Mat(contour));
//	minRectable =  cv::boundingRect(cv::Mat(contour));
//	cv::ellipse(dst, minEllipse, colorEllipse, 2, 8 );
//	cv::rectangle(dst,minEllipse.boundingRect(),2,1);


	cv::Point2f vertices[4];
	cv::Point2f axis[4];
	minEllipse.points(vertices);
	for (int i = 0; i < 4; i++)
	{
//		cv::line(dst, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));
		axis[i].x = (vertices[i].x + vertices[(i + 1) % 4].x) / 2;
		axis[i].y = (vertices[i].y + vertices[(i + 1) % 4].y) / 2;
	}

//	cv::line(dst, axis[0], axis[2], cv::Scalar(255,0,0));
	cv::line(dst, axis[1], axis[3], cv::Scalar(255, 0, 0));
	index++;
}


void drawBoxes(std::vector<cv::Rect>& boxes, cv::Mat& dst, const float factor)
{
	size_t N = boxes.size();
	for (size_t n = 0; n < N; ++n)
	{
		cv::Rect box = boxes[n];
		//redefine upper left corner
		box.y      += box.height * (1 - factor) * 0.5;
		box.x      += box.width * (1 - factor) * 0.5;
		box.height *= factor;
		box.width  *= factor;
		cv::rectangle(dst, box, cv::Scalar(255, 0, 0), 2);
	}
}



bool isFish(cv::linemod::Match& mHead, cv::linemod::Match& mTail, int num_modalities, float threshold, cv::Mat& dst,
	          cv::Ptr<cv::linemod::Detector>& detHead,
		      cv::Ptr<cv::linemod::Detector>& detTail,
		      ConfigFile &config)
{
	//getting templates
	const std::vector<cv::linemod::Template>& templatesHead = detHead->getTemplates(mHead.class_id, mHead.template_id);
	const std::vector<cv::linemod::Template>& templatesTail = detTail->getTemplates(mTail.class_id, mTail.template_id);

	cv::Point(mHead.x, mHead.y);
//	detHead->getT(0);

	cv::RotatedRect minElipse[2];
	cv::Rect minRect[2];
	cv::Point centRect[2];
	cv::vector<cv::Point> contourHead, contourTail;
	cv::vector<cv::vector<cv::Point> > contours;
	cv::Point ptin, ptout;

	// save templates as contours for analisis
	for (int m = 0; m < num_modalities; ++m)
	{
		for (int i = 0; i < (int)templatesHead[m].features.size(); ++i)
		{
			cv::linemod::Feature f = templatesHead[m].features[i];
			cv::Point pt(f.x + mHead.x, f.y + mHead.y);
			contourHead.push_back(pt);
		}
		for (int i = 0; i < (int)templatesTail[m].features.size(); ++i)
		{
			cv::linemod::Feature f = templatesTail[m].features[i];
			cv::Point pt(f.x + mTail.x, f.y + mTail.y);
			contourTail.push_back(pt);
		}
	}

	//save contours in a vector
	contours.push_back(contourHead);
	contours.push_back(contourTail);

	//save axis of elipse
	cv::Point2f axis[2][4];

	//find minimum ellipse and boundingbox for template draw it
	for (int i = 0; i < 2; ++i)
	{
		minElipse[i] = cv::fitEllipse(cv::Mat(contours[i]));
		minRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		centRect[i] = cv::Point(minRect[i].x + minRect[i].width/2, minRect[i].y + minRect[i].height/2);
		//	cv::ellipse(dst, minEllipse, colorEllipse, 2, 8 );
		//	cv::rectangle(dst,minEllipse.boundingRect(),2,1);
		cv::Point2f vertices[4];
		minElipse[i].points(vertices);

		//set axis of elipse
		for (int j = 0; j < 4; j++)
		{
			axis[i][j].x = (vertices[j].x + vertices[(j+1)%4].x)/2;
			axis[i][j].y = (vertices[j].y + vertices[(j+1)%4].y)/2;
		}

	}

	double dist = cv::norm(cv::Mat(centRect[0]), cv::Mat(centRect[1]), cv::NORM_L2);
//	double lengthHead = cv::norm(cv::Mat(axis[0][1]), cv::Mat(axis[0][3]), cv::NORM_L2);
//	double lengthTail = cv::norm(cv::Mat(axis[1][3]), cv::Mat(axis[1][3]), cv::NORM_L2);
	double lengthHead = minRect[0].width;
	double lengthTail = minRect[1].width;
	double meanSize = (lengthHead + lengthTail)/2;

//	cv::vector<cv::Point> ptHead = extendAxis( contourHead, axis[0][0], axis[0][2], centRect[0], dst);
//	cv::vector<cv::Point> ptTail = extendAxis( contourHead, axis[1][0], axis[1][2], centRect[0], dst);

    //find line between to points
    cv::vector<cv::Point3f> ptHead_h;
    cv::vector<cv::Point3f> ptTail_h;
//    for(int i = 0; i < 2; ++i)
//    {
    	ptHead_h.push_back(cv::Point3f(axis[0][1].x, axis[0][1].y,1));
    	ptHead_h.push_back(cv::Point3f(axis[0][3].x, axis[0][3].y,1));
    	ptTail_h.push_back(cv::Point3f(axis[1][1].x, axis[1][1].y,1));
    	ptTail_h.push_back(cv::Point3f(axis[1][3].x, axis[1][3].y,1));
//    }

    cv::Point3f lnHead_h = ptHead_h[0].cross(ptHead_h[1]);
    cv::Point3f lnTail_h = ptTail_h[0].cross(ptTail_h[1]);

    cv::Point3d crosspt_h = lnHead_h.cross(lnTail_h);
    cv::Point2f crosspt = cv::Point2f(((crosspt_h.x)/crosspt_h.z), ((crosspt_h.y)/crosspt_h.z));

    // find extrem point from template
    cv::vector<cv::Point> extrHead_h;
    cv::vector<cv::Point> extrTail_h;
    int sideHead = findTemplateSide(contourHead, minElipse[0], minRect[0], dst, extrHead_h);
    int sideTail = findTemplateSide(contourTail, minElipse[1], minRect[1], dst, extrTail_h);


    float omega[4] = {	config.read<float>("omega0"), //Disntance between center of boundingboxes templates
    					config.read<float>("omega1"), //Angle between templates
    					config.read<float>("omega2"), //Distance between Upper extrem templates
    					config.read<float>("omega2")}; //Distance between Lower extrem templates
    float f[4];
    float F;
    float alpha[3] = {	config.read<float>("alpha0"), //Similarity Head Templates
    					config.read<float>("alpha1"), //Similarity Tail Templates
    					config.read<float>("alpha2")}; //Closeness between templates
    float Final;
    float angleL   = config.read<float>("angleL");   // angle between template LEFT
    float angleR   = config.read<float>("angleR");  // angle between template RIGHT
    float dist_k   = config.read<float>("dist_k");   // ref distance between template corners
    int   scale    = config.read<int>("scale");      //factor to define how many time the size of the template to check distance between template
    int   overlap_k  = config.read<int>("overlap_k"); //Ovelap between templates
    int   overlap[2];

    bool isFish    = false;
    bool sideCheck = false;

    //Calculate distance and angle between template extrem
    cv::Vec2d speed;
    double teta[2];
    double distT_0 = cv::norm(cv::Mat(extrHead_h[0]), cv::Mat(extrTail_h[0]), cv::NORM_L2);
    double distT_1 = cv::norm(cv::Mat(extrHead_h[1]), cv::Mat(extrTail_h[1]), cv::NORM_L2);

    std::ostringstream direction;

    if ((sideHead == LEFT && sideTail == RIGHT) && (centRect[0].x <= centRect[1].x))
    {
    	overlap[0] =  extrTail_h[0].x - extrHead_h[0].x;
    	overlap[1] =  extrHead_h[1].x - extrTail_h[1].x;
    	if((overlap[0] >= -1*overlap_k) && (overlap[1] >= -1*overlap_k))
    	{

    		springEq(extrTail_h[0], extrHead_h[0], speed, teta[0]);
    		springEq(extrTail_h[1], extrHead_h[1], speed, teta[1]);
    		f[0] = dist < scale*meanSize? ((scale*meanSize - dist)/(scale*meanSize)) : 0;
    		f[1] = (abs(teta[0]) <= angleL) && (abs(teta[1]) <= angleL)? ((angleL - std::max(abs(teta[0]),abs(teta[1])))/angleL): 0;
    		f[2] = (distT_0 <= dist_k)? ((dist_k- distT_0)/dist_k): 0;
    		f[3] = (distT_1 <= dist_k)? ((dist_k - distT_1)/dist_k): 0;

    		direction << "LEFTtoRIGHT";
    		sideCheck = true;
    	}
    }
    else if ((sideHead == RIGHT && sideTail == LEFT) && (centRect[0].x >= centRect[1].x))
    {
    	overlap[0] = extrHead_h[0].x - extrTail_h[0].x;
    	overlap[1] = extrHead_h[1].x - extrTail_h[1].x;
    	if((overlap[0] >= -1*overlap_k) && (overlap[1] >= -1*overlap_k))
    	{

    		springEq(extrHead_h[0], extrTail_h[0], speed, teta[0]);
    		springEq(extrHead_h[1], extrTail_h[1], speed, teta[1]);
    		f[0] = dist < scale*meanSize? ((scale*meanSize-dist)/(scale*meanSize)) : 0;
    		f[1] = (abs(teta[0]) <= angleL) && (abs(teta[1]) <= angleL)? ((angleL - std::max(abs(teta[0]),abs(teta[1])))/angleL): 0;
    		f[2] = (distT_0 <= dist_k)? ((dist_k - distT_0)/dist_k): 0;
    		f[3] = (distT_1 <= dist_k)? ((dist_k - distT_1)/dist_k): 0;

    		direction << "RIGHTtoLEFT";
    		sideCheck = true;
		}
	}



    if(sideCheck)
    {

    	//Calculating compare function
    	F     = omega[0]*f[0] + omega[1]*f[1] + omega[2]*f[2] + omega[3]*f[3];
    	Final = alpha[0]*mHead.similarity + alpha[1]*mTail.similarity + alpha[2]*100*F;


    	if(Final >= threshold)
    	{
    		std::cout << "-----------------------********************--------------------" << std::endl;
    		std::cout << "class Head = " << mHead.class_id <<  " -- ///// -- ";
    		std::cout << "class Tail = " << mTail.class_id << std::endl;

    		std::cout << "Fish Direction = " << direction.str()  << std::endl;
    		std::cout << "overlap 0 = " << overlap[0] <<  " -- ///// -- ";
    		std::cout << "overlap 1 = " << overlap[1] << std::endl;

    		for(int i = 0; i < 4; ++i)
    			std::cout << "func [" << i << "] = " << f[i]<< " ; ";
    		std::cout << std::endl;

    		std::cout << "Dist = " << dist <<  " -- ///// -- ";
    		std::cout << "meanSize = " << meanSize << std::endl;
    		std::cout << "teta Up = " << teta[0] << std::endl;
    		std::cout << "teta Down = " << teta[1] << std::endl;

    		std::cout << "threshold = " << threshold << std::endl;
    		std::cout << "Function = "  << F*100     << std::endl;
    		std::cout << "Final = "     << Final     << std::endl;
    		std::cout << "-----------------------********************--------------------" << std::endl;

    		cv::line(dst, centRect[0]  , centRect[1]  , cv::Scalar(255,0,0));
    		cv::line(dst, extrHead_h[0], extrTail_h[0], cv::Scalar(0,255,0));
    		cv::line(dst, extrHead_h[1], extrTail_h[1], cv::Scalar(0,255,0));

    		cv::circle(dst, crosspt      , 4, cv::Scalar(0,0,255)  , 2,  1);
    		cv::circle(dst, extrHead_h[0], 5, cv::Scalar(0,50,255) , 2,  1);
    		cv::circle(dst, extrHead_h[1], 5, cv::Scalar(0,100,255), 2,  1);
    		cv::circle(dst, extrTail_h[0], 5, cv::Scalar(255,50,0) , 2,  1);
    		cv::circle(dst, extrTail_h[1], 5, cv::Scalar(255,100,0), 2,  1);
    		cv::circle(dst, centRect[0]  , 3, cv::Scalar(0,0,0)    , 2, -1);
    		cv::circle(dst, centRect[1]  , 3, cv::Scalar(0,0,0)    , 2, -1);

    		std::ostringstream thres;
    		thres << "Threshold = " << (int)threshold;
    		cv::putText(dst, thres.str(), cv::Point(30,15),
    				cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, CV_AA);
    		std::ostringstream similarity;
    		similarity << "Func = " << Final;
    		cv::putText(dst, similarity.str(), cv::Point(30,30),
    				cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, CV_AA);

    		isFish = true;
    	}
    }

//	cv::namedWindow("isFish");
//	cv::imshow("isFish", dst);
	return isFish;
}



int findTemplateSide(cv::vector<cv::Point> contour, cv::RotatedRect elipse, cv::Rect rect, cv::Mat& dst, cv::vector<cv::Point>& extrem)
{


	cv::Mat mask = cv::Mat::zeros(dst.size(), CV_8UC3 );
	cv::Mat mask1 = cv::Mat::zeros(dst.size(), CV_8UC1 );
	cv::vector<cv::Point> extendline;

    extrem = sortExtremContour( contour, elipse, rect, dst);

    cv::vector<cv::Point>::const_iterator it;
    for( it = contour.begin(); it != contour.end()-1; ++it )
    {
    	cv::circle( mask1, *it, 2, cv::Scalar(255), 1, -1);
    }

    cv::circle( mask1, extrem[0], 2, cv::Scalar(100), 3, -1);
    cv::circle( mask1, extrem[1], 2, cv::Scalar(100), 3, -1);
    cv::circle( mask1, cv::Point(rect.x + rect.width/2, rect.y + rect.height/2), 2, cv::Scalar(200), 3, -1);

//    cv::namedWindow("interception");
//    cv::imshow("interception", mask1);

    if ((extrem[0].x > (rect.x + rect.width/2)) && extrem[1].x > (rect.x + rect.width/2))
    {
    	return LEFT;
    }
    else if ((extrem[0].x <= (rect.x + rect.width/2)) && extrem[1].x <= (rect.x + rect.width/2))
    {
    	return RIGHT;
    }

	return NONE;
}


//generate mask and crop image
 void subtractMask(cv::vector<cv::Mat>& source, cv::Mat& mask, cv::vector<cv::Mat>& masked, int& maskside)
{

	//Set size of mask
	mask = cv::Mat(source[0].size(), CV_8UC1, cv::Scalar(0));
	cv::Mat color = cv::Mat(source[0].size(), source[0].type(), cv::Scalar(255,255,255));
	cv::Mat temp = 255*masked[1].clone();

	switch (maskside)
	{
		//sent mask to left
		case LEFT:
			temp.copyTo(mask.colRange(0, masked[1].cols)
					.rowRange((mask.rows - masked[1].rows)/2, (mask.rows + masked[1].rows)/2));
			masked[0].copyTo(color.colRange(0, masked[0].cols)
					.rowRange((color.rows - masked[0].rows)/2, (color.rows + masked[0].rows)/2));
			break;
		//sent mask to right
		case RIGHT:
			temp.copyTo(mask.colRange(mask.cols - temp.cols, mask.cols)
					.rowRange((mask.rows - temp.rows)/2, (mask.rows + temp.rows)/2));
			masked[0].copyTo(color.colRange(source[0].cols - masked[0].cols, color.cols)
					.rowRange((color.rows - masked[0].rows)/2, (color.rows + masked[0].rows)/2));
			break;
		case NONE:
			mask = cv::Mat(source[0].size(), CV_8UC1, cv::Scalar(0));
			color = cv::Mat(source[0].size(), source[0].type(), cv::Scalar(255,255,255));
			break;
		default:
			break;
	}

	source[0] = color.clone();

	//visualize extracted foreground
	cv::namedWindow("subtractmask");
	cv::imshow("subtractmask", source[0]);
}


//sort a contour clockwise staring from center of bounding box
 cv::vector<cv::Point> sortExtremContour(cv::vector<cv::Point>& contour, cv::RotatedRect& elipse, cv::Rect& rect, cv::Mat& dst)
 {
	 // Draw black contours on a white image
//	 cv::Mat result(dst.size(),CV_8U,cv::Scalar(255));

	 //sort contour
	 cv::vector<cv::Point>::iterator it;
	 bool swaped = true;
	 while (swaped)
	 {
		 swaped = false;
		 for( it = contour.begin(); it != contour.end()-1; ++it )
		 {
			 cv::Point cur_pt = *it;
			 cv::Point pos_pt = *(it+1);
			 if(less(cur_pt, pos_pt, cv::Point(rect.x + rect.width/2, rect.y + rect.height/2)))
			 {
				 *it= pos_pt;
				 *(it+1) = cur_pt;
				 swaped = true;
			 }
		 }
	 }

	 //Find opening of contour
	 double maxdist = 4;
	 cv::Point point[2];
	 for( it = contour.begin(); it != contour.end()-1; ++it )
	 {
//		 cv::circle( result, *it, 2, cv::Scalar(0), 1, -1);
		 double dist = cv::norm(cv::Mat(*(it)), cv::Mat(*(it+1)), cv::NORM_L2);
		 if(dist > maxdist)
		 {
			 point[0] = *(it);
			 point[1] = *(it+1);
			 maxdist = dist;
		 }
	 }

	 cv::vector<cv::Point> extrem;

	 //check position of point
	 if(point[0].y >= point[1].y)
	 {
		 //set point 0 as upper
		 extrem.push_back(point[0]);
		 extrem.push_back(point[1]);
	 }
	 else if (point[0].y < point[1].y)
	 {
		 //set point 1 as upper
		 extrem.push_back(point[1]);
		 extrem.push_back(point[0]);
	 }

//	 cv::circle( result, extrem[0], 2, cv::Scalar(150), 2, -1);
//	 cv::circle( result, extrem[1], 2, cv::Scalar(150), 2, -1);
//	 cv::namedWindow("contori");
//	 cv::imshow("contori", result);
	 return extrem;
 }


 //define if a point in contour is more to left of center
 bool less(cv::Point a, cv::Point b, cv::Point center)
 {
     if (a.x-center.x >= 0 && b.x-center.x < 0)
         return true;
     if (a.x-center.x == 0 && b.x-center.x == 0) {
         if (a.y-center.y >= 0 || b.y-center.y >= 0)
             return a.y > b.y;
         return b.y > a.y;
     }

     // compute the cross product of vectors (center -> a) x (center -> b)
     int det = (a.x-center.x) * (b.y-center.y) - (b.x - center.x) * (a.y - center.y);
     if (det < 0)
         return true;
     if (det > 0)
         return false;

     // points a and b are on the same line from the center
     // check which point is closer to the center
     int d1 = (a.x-center.x) * (a.x-center.x) + (a.y-center.y) * (a.y-center.y);
     int d2 = (b.x-center.x) * (b.x-center.x) + (b.y-center.y) * (b.y-center.y);
     return d1 > d2;
 }


 //Calculate spring equation with acceleration equal 0
double springEq(cv::Point pt_1, cv::Point pt_2, cv::Vec2d& speed, double& teta)
{
	double magSpeed;
	const double k = 1; // spring const
	const double b = 1; // damping const
	const double R = 10; // rest lenght spring

	double L = cv::norm(cv::Mat(pt_1), cv::Mat(pt_2), cv::NORM_L2);
	double S = L - R;
	double sin_teta = (pt_2.x - pt_1.x) / L;
	double cos_teta = (pt_2.y - pt_1.y) / L;

	double tetarad = (pt_2.x - pt_1.x) == 0 ? 0 : atan((pt_2.y - pt_1.y) / (pt_2.x - pt_1.x));
//	 double tetarad = atan2 ((pt_2.y - pt_1.y),(pt_2.x - pt_1.x));
	teta = tetarad * 180 / cv_pi;
//	 printf ("The arc tangent for (x=%f, y=%f) is %f degrees\n", (pt_2.x - pt_1.x), (pt_2.x - pt_1.x), tetadeg );

	speed.val[0] = -1 * (k / b) * S * sin_teta;
	speed.val[1] = -1 * (k / b) * S * cos_teta;

	magSpeed = cv::norm(speed, cv::NORM_L2);

	return magSpeed;
}


 cv::Mat display3(std::vector<cv::Mat>& images)
 {

     size_t N = images.size();
     const int rows = images[0].rows;
     const int cols = images[0].cols;
     cv::Mat canvas = ::cv::Mat::zeros(rows, N*cols, CV_8UC3);
     for (size_t n = 0; n < N; ++n)
     {
         cv::Mat imagesColRange = canvas.colRange(n*cols,(n+1)*cols);
         images[n].copyTo(imagesColRange);
     }

     return canvas;
 }




