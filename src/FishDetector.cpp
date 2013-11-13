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

#include <boost/filesystem.hpp>
#include <cstdio>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
//#ifdef WITH_MATLABIO
	#include "MatlabIOModel.hpp"
//#endif
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"

const double cv_pi = 3.141592653589;

enum sendTo
{
	NONE  = 0,
	RIGHT = 1,
	LEFT  = 2
};


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

bool isFish(cv::linemod::Match& mHead, cv::linemod::Match& mTail, int num_modalities, float threshold, cv::Mat& dst,
		      cv::Ptr<cv::linemod::Detector>& detHead,
			  cv::Ptr<cv::linemod::Detector>& detTail);

cv::Mat displayQuantized(const cv::Mat& quantized);

//Calculate spring equation with acceleration equal 0
double springEq(cv::Point pt_1, cv::Point pt_2, cv::Vec2d& speed, double& teta);

//Prepare mask for training linemod
void subtractMask( cv::vector<cv::Mat>& color, cv::Mat& mask, cv::vector<cv::Mat>& masked, int& maskside);

int findTemplateSide(cv::vector<cv::Point> contour, cv::RotatedRect elipse, cv::Rect rect, cv::Mat& dst, cv::vector<cv::Point>& extrem);

cv::vector<cv::Point> sortExtremContour(cv::vector<cv::Point>& contour, cv::RotatedRect& elipse, cv::Rect& rect, cv::Mat& dst);

bool less(cv::Point a, cv::Point b, cv::Point center);

static void help()
{
  printf("Usage: line2ddetector -tail linemod_tail.yml -head linemod_head.yml -fn fishlist.yaml \n\n"
         "Load the trained linemod detector and fishlist as show above,\n"
         "Keys:\n"
         "\t h     -- This help page\n"
         "\t m     -- Toggle printing match result\n"
         "\t t     -- Toggle printing timings\n"
         "\t k     -- Toggle shows matches\n"
         "\t o     -- Toggle delaying between draw matches\n"
         "\t w     -- Write learned templates to disk\n"
         "\t [d/u] -- Adjust matching threshold: '[' down,  ']' up\n"
         "\t [s/z] -- Adjust matching threshold: '[' down,  ']' up\n"
         "\t [f/b] -- Move frame : '[' forward,  ']' backward\n"
         "\t p     -- Pause frames \n"
         "\t q     -- Quit\n\n");
}

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
  cv::Ptr<cv::linemod::Detector> detector = new cv::linemod::Detector;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  detector->read(fs.root());

  cv::FileNode fn = fs["classes"];
  for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

static void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<std::string> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}

//hide the local functions in an unnamed namespace
namespace
{
void helpread(char** av)
{
	std::cout
			<< "\nThis program gets you started being able to read images from a list in a file\n"
					"Usage:\n./" << av[0] << " image_list.yaml\n"
			<< "\tThis is a starter sample, to get you up and going in a copy pasta fashion.\n"
			<< "\tThe program reads in an list of images from a yaml or xml file and displays\n"
			<< "one at a time\n"
			<< "\tTry running imagelist_creator to generate a list of images.\n"
					"Using OpenCV version %s\n" << CV_VERSION << "\n"
				"\t-lm follow by template file name.\n"
				"\t-fn follow by list image filename.\n"	<< std::endl;
}

bool readStringList(const std::string& filename, std::vector<std::string>& l)
{
	l.resize(0);
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;
	cv::FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != cv::FileNode::SEQ)
		return false;
	cv::FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((std::string) *it);
	return true;
}
}

// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
	static void start(const std::string& a_img_name)
	{
		cvSetMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
	}
	static int event(void)
	{
		int l_event = m_event;
		m_event = -1;
		return l_event;
	}
	static int x(void)
	{
		return m_x;
	}
	static int y(void)
	{
		return m_y;
	}

private:
	static void cv_on_mouse(int a_event, int a_x, int a_y, int, void *)
	{
		m_event = a_event;
		m_x = a_x;
		m_y = a_y;
	}

	static int m_event;
	static int m_x;
	static int m_y;
};

int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;

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



int main(int argc, char * argv[])
{
  // Various settings and flags
  bool show_match_result = true;
  bool show_timings = false;
  bool show_matches = true;
  bool learn_online = false;
  bool savedlmHead = false;
  bool savedlmTail = false;
  bool delaying = false;
  bool press_key_a = false;
//  bool maskloaded = false;
//  int maskside = NONE;
  char key;
  cv::vector<int> num_classes;
//  int matching_threshold = 88;
  int matching_threshold_head = 87; //91
  int matching_threshold_tail = 87; //89
  int matching_threshold_fish = 76;
  int nrCandidate = 3;
  /// @todo Keys for changing these?
  cv::Size roi_size(100, 100);
//  int learning_lower_bound = 90;
//  int learning_upper_bound = 95;

  int time = 0;
  bool pause = false;

  std::string linemodHead;
  std::string linemodTail;
  std::string partBasedModel;
  std::string filename;

  // Initialize LINEMOD data structures
  cv::Ptr<cv::linemod::Detector> detHead;
  cv::Ptr<cv::linemod::Detector> detTail;

  for( int i = 1; i < argc; i++ )
  {
	  if( std::string(argv[i]) == "-tail" )
	  {
		  linemodTail = argv[++i];
		  savedlmTail = true;
	  }
	  if( std::string(argv[i]) == "-head" )
	  {
		  linemodHead = argv[++i];
		  savedlmHead = true;
	  }
	  if( std::string(argv[i]) == "-model" )
	  {
		  partBasedModel = argv[++i];
	  }
	  else if( std::string(argv[i]) == "-fn" )
		  filename = argv[++i];
	  else if( std::string(argv[i]) == "--help" )
	  {
		  helpread(&argv[i]);
		  return 0;
	  }
	  else if( argv[i][0] == '-' )
	  {
		  std::cout << "invalid option " << argv[i] << std::endl;
		  return 0;
	  }
//	  else
//	  {
//		  std::cout << "invalid option " << argv[i] << endl;
//		  return 0;
//	  }

  }

  if( !savedlmHead && !savedlmTail )
  {
       linemodHead = "linemod_head.yml";
       linemodTail = "linemod_tail.yml";
       detHead = cv::linemod::getDefaultLINE();
       detTail = cv::linemod::getDefaultLINE();
  }
  else
  {
	//Loading trained detector
    detHead = readLinemod(linemodHead);
    detTail = readLinemod(linemodTail);

    time = 10;

    cv::vector<std::vector<std::string> > ids;
    ids.push_back(detHead->classIds());
    ids.push_back(detTail->classIds());

    num_classes.push_back(detHead->numClasses());
    num_classes.push_back(detTail->numClasses());

    printf("Loaded %s with %d classes and %d templates\n",
           linemodHead.c_str(), num_classes[0], detHead->numTemplates());
    printf("Loaded %s with %d classes and %d templates\n",
           linemodTail.c_str(), num_classes[1], detTail->numTemplates());
    if (!ids[0].empty())
    {
      printf("ClassHead ids:\n");
      std::copy(ids[0].begin(), ids[0].end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
    if (!ids[1].empty())
    {
      printf("ClassTail ids:\n");
      std::copy(ids[1].begin(), ids[1].end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
  }

  // determine the type of model to read
  boost::scoped_ptr<Model> model;
  std::string ext = boost::filesystem::path(partBasedModel.c_str()).extension().string();
  if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0) {
	  model.reset(new FileStorageModel);
  }
//#ifdef WITH_MATLABIO
  else if (ext.compare(".mat") == 0) {
	  model.reset(new MatlabIOModel);
  }
//#endif
  else {
	  printf("Unsupported model format: %s\n", ext.c_str());
	  exit(-2);
  }
  bool ok = model->deserialize(partBasedModel.c_str());
  if (!ok) {
	  printf("Error deserializing file\n");
	  exit(-3);
  }

  // create the PartsBasedDetector and distribute the model parameters
  PartsBasedDetector<float> pbd;
  pbd.distributeModel(*model);


  // Timers
  Timer extract_timer;
  Timer match_timer;

  // Initialize HighGUI
  help();
  cv::namedWindow("color");
//  cv::namedWindow("normals");
  cv::namedWindow("connected");
  Mouse::start("color");


  int num_modalities = (int)detHead->getModalities().size();
  printf("num_modalities %d\n", num_modalities);

  //load image adress
  std::vector<std::string> imagelist;
  ok = readStringList(filename, imagelist);
  if(!ok || imagelist.empty())
  {
      std::cout << "can not open " << filename << " or the string list is empty" << std::endl;
//      helpread();
      return 0;
  }

   // Main loop
  cv::Mat color;

//Run over image list
  for (unsigned int  j = 0; j < imagelist.size(); j++)
  {

//	  printf("begin cicle j = %d \n", j);

	  //Read image from imagelist
	  cv::Mat test = cv::imread(imagelist[j], CV_LOAD_IMAGE_COLOR); // do grayscale processing?
	  std::cout << "///" << imagelist[j] << "///" << std::endl;
	  cv::Mat_<float> depth; //Added for PartBasesModel

	  //Resize image to limite process time
	  if((test.rows > 480) || (test.cols > 640) )
	  {
		  cv::resize(test, test, cv::Size(640,480));
	  }

	  color = test.clone();

	  std::vector<cv::Mat> sources;
	  sources.push_back(color);
	  cv::Mat display = color.clone();
	  cv::Mat display2 = color.clone();

	  // detect potential candidates in the image
	  std::vector<Candidate> candidates;
	  pbd.detect(color, depth, candidates);
	  printf("Number of candidates: %ld\n", candidates.size());


	  // display the best candidates
	  Visualize visualize(model->name());
	  SearchSpacePruning<float> ssp;
	  cv::Mat canvas;
	  if (candidates.size() > 0) {
		  Candidate::sort(candidates);
		  Candidate::nonMaximaSuppression(color, candidates, 0.3);
		  visualize.candidates(color, candidates, nrCandidate, canvas, true);
		  std::cout << "candidate = " << candidates.size() << std::endl;
		  visualize.image(canvas);
//		  cv::waitKey();
	  }



	  // Perform matching
	  std::vector<cv::linemod::Match>  matchesHead;
	  std::vector<cv::linemod::Match>  matchesTail;
	  std::vector<std::string>  class_idsHead;
	  std::vector<std::string>  class_idsTail;
	  std::vector<cv::Mat>  quantized_imagesHead;
	  std::vector<cv::Mat>  quantized_imagesTail;
	  match_timer.start();
	  detHead->match(sources, (float)matching_threshold_head, matchesHead, class_idsHead, quantized_imagesHead);
	  detTail->match(sources, (float)matching_threshold_tail, matchesTail, class_idsTail, quantized_imagesTail);

	  cv::vector<std::vector<cv::linemod::Match> > matches;
	  matches.push_back(matchesHead);
	  matches.push_back(matchesTail);
	  cv::vector<std::vector<std::string> > class_ids;
	  class_ids.push_back(class_idsHead);
	  class_ids.push_back(class_idsTail);
	  cv::vector<std::vector<cv::Mat> > quantized_images;
	  quantized_images.push_back(quantized_imagesHead);
	  quantized_images.push_back(quantized_imagesTail);

//	  cv::Mat colored = displayQuantized(quantized_imagesHead[0]);
	  match_timer.stop();

	  cv::vector<int> classes_visited;
	  classes_visited.push_back(0);
	  classes_visited.push_back(0);
	  //    cv::vector<std::set<std::string> > visited;
	  std::set<std::string>  visitedHead;
	  std::set<std::string>  visitedTail;

	  if(show_matches)
	  {
		  for (int k = 0; k < 2; ++k)
		  {
//			  for (int i = 0; (i < (int)matches[k].size()) && (classes_visited[k] < num_classes[k]); ++i)
			  for (int i = 0; (i < (int)matches[k].size()); ++i)
			  {
				  cv::linemod::Match m = matches[k][i];

				  if (k == 0)
				  {
					  if (visitedHead.insert(m.class_id).second)
					  {
						  ++classes_visited[k];

						  if (show_match_result)
						  {
							  printf("SimilarityHead: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
									  m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
						  }

						  // Draw matching template
						  const std::vector<cv::linemod::Template>& templatesHead = detHead->getTemplates(m.class_id, m.template_id);
						  drawResponse(templatesHead, num_modalities, display, cv::Point(m.x, m.y), detHead->getT(0));
						  cv::putText(display, m.class_id, cv::Point(m.x,m.y),
								  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,0,250), 1, CV_AA);
					  }
				  }
				  else if (k == 1)
				  {
					  if (visitedTail.insert(m.class_id).second)
					  {
						  ++classes_visited[k];

						  if (show_match_result)
						  {
							  printf("SimilarityTail: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
									  m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
						  }

						  // Draw matching template
						  const std::vector<cv::linemod::Template>& templatesTail = detTail->getTemplates(m.class_id, m.template_id);
						  drawResponse(templatesTail, num_modalities, display, cv::Point(m.x, m.y), detTail->getT(0));
						  cv::putText(display, m.class_id, cv::Point(m.x,m.y),
								  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,250), 1, CV_AA);
					  }
				  }
			  }
			  //Reinit counter
			  classes_visited[0] = 0;
			  classes_visited[1] = 0;
		  }
	  }

	  //Reinit counter
	  classes_visited[0] = 0;
	  classes_visited[1] = 0;

	  // go through all head matches, and connecting  head-tail
//	  for (int i = 0; (i < (int)matches[0].size()) && (classes_visited[0] < num_classes[0]); ++i)
	  for (int i = 0; (i < (int)matches[0].size()); ++i)
	  {
		  ++classes_visited[0];
		  //go through all Tail matches
//		  for (int j = 0; (j < (int)matches[1].size()) && (classes_visited[1] < num_classes[1]); ++j)
		  for (int j = 0; (j < (int)matches[1].size()); ++j)
		  {
			  ++classes_visited[1];
			  //loading matches
			  cv::linemod::Match mHead = matches[0][i];
			  cv::linemod::Match mTail = matches[1][j];


			  //getting templates
			  const std::vector<cv::linemod::Template>& templatesHead = detHead->getTemplates(mHead.class_id, mHead.template_id);
			  const std::vector<cv::linemod::Template>& templatesTail = detTail->getTemplates(mTail.class_id, mTail.template_id);


			  bool Fish = isFish(mHead, mTail, num_modalities, (float)matching_threshold_fish, display2, detHead, detTail);


			  if(Fish)
			  {

				  drawResponse(templatesHead, num_modalities, display2, cv::Point(mHead.x, mHead.y), detHead->getT(0));
				  drawResponse(templatesTail, num_modalities, display2, cv::Point(mTail.x, mTail.y), detTail->getT(0));

				  cv::putText(display2, mHead.class_id, cv::Point(mHead.x,mHead.y),
						  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,250), 1, CV_AA);
				  cv::putText(display2, mTail.class_id, cv::Point(mTail.x,mTail.y),
						  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,0,250), 1, CV_AA);


				  printf("SimilarityHead: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
						  mHead.similarity, mHead.x, mHead.y, mHead.class_id.c_str(), mHead.template_id);

				  printf("SimilarityTail: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
						  mTail.similarity, mTail.x, mTail.y, mTail.class_id.c_str(), mTail.template_id);

				  if(delaying)
				  {
					  cv::imshow("connected", display2);
					  cv::waitKey(200);
				  }
			  }
//			  cv::waitKey();
		  }
		  //Reinit counter
		  classes_visited[1] = 0;
	  }
	  //Reinit counter
	  classes_visited[0] = 0;


    if (show_match_result && matches.empty())
      printf("No matches found...\n");
    if (show_timings)
    {
      printf("Training: %.2fs\n", extract_timer.time());
      printf("Matching: %.2fs\n", match_timer.time());
    }
    if (show_match_result || show_timings)
      printf("------------------------------------------------------------\n");

    cv::imshow("color", display);
//    cv::imshow("normals", colored);
    cv::imshow("connected",display2);

    cv::FileStorage fs;

    key = (char)cvWaitKey(time);

    if(j == imagelist.size()-2)
    	key = 'w';

    if(j == (imagelist.size()-1))
    	j = 0;

    if( key == 'q' )
        break;


    switch (key)
    {
      case 'h':
        help();
        break;
      case 'm':
        // toggle printing match result
        show_match_result = !show_match_result;
        printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
        break;
      case 't':
        // toggle printing timings
        show_timings = !show_timings;
        printf("Show timings %s\n", show_timings ? "ON" : "OFF");
        break;
      case 'k':
        // toggle show matches
        show_matches = !show_matches;
        printf("Show matches %s\n", show_matches ? "ON" : "OFF");
        break;
      case 'o':
        // toggle delaying drawfish
        delaying = !delaying;
        printf("Show delaying %s\n", delaying ? "ON" : "OFF");
        break;
      case 'l':
        // toggle online learning
        learn_online = !learn_online;
        if(learn_online)
        	time = 10;
        else
        	time = 0;

        printf("Online learning %s\n", learn_online ? "ON" : "OFF");
        break;
      case 'u':
        // decrement threshold head
        matching_threshold_head = std::max(matching_threshold_head - 1, -100);
        printf("New threshold head: %d\n", matching_threshold_head);
        // decrement threshold tail
        matching_threshold_tail = std::max(matching_threshold_tail - 1, -100);
        printf("New threshold tail: %d\n", matching_threshold_tail);
        break;
      case 'd':
        // increment threshold head
        matching_threshold_head = std::min(matching_threshold_head + 1, +100);
        printf("New threshold: %d\n", matching_threshold_head);
        // increment threshold tail
        matching_threshold_tail = std::min(matching_threshold_tail + 1, +100);
        printf("New threshold tail: %d\n", matching_threshold_tail);
        break;
      case 'z':
        // decrement threshold fish
//        matching_threshold_fish = std::max(matching_threshold_fish - 0.1, (double)(-1));
        matching_threshold_fish = std::max(matching_threshold_fish - 1, -100);
        printf("New threshold head: %d\n", matching_threshold_fish);
        break;
      case 's':
        // increment threshold fish
//        matching_threshold_fish = std::min(matching_threshold_fish + 0.1, (double)(+1));
        matching_threshold_fish = std::min(matching_threshold_fish + 1, +100);
        printf("New threshold: %d\n", matching_threshold_fish);
        break;
      case 'w':
        // write model to disk
        writeLinemod(detHead, linemodHead);
        writeLinemod(detTail, linemodTail);
        printf("Wrote detector and templates to %s\n", linemodHead.c_str());
        printf("Wrote detector and templates to %s\n", linemodTail.c_str());
        break;
    	  break;
      case 'a':
        // silmulate Right botton down
        press_key_a = true;
        printf("Press key a \n");
        break;
      case 'p':
        // pause sytem
        pause = !pause;
        if(pause)
        	time = 10;
        else
        	time = 0;
        printf("Press key p \n");
        break;
      case 'b':
        // backward one frame
        if(!pause)
        	j = j - 2;
        printf("Press key b j = %d \n", j);
        printf("Pause = %d \n", pause);
        break;
      case 'f':
        // forward one frame
//        if(!pause)
//        	j = j ;
        printf("Press key b j = %d \n", j);
        printf("Pause = %d \n", pause);
        break;
      default:
    	  break;
    }
  }
  return 0;
}


static void reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f)
{
  real.resize(proj.size());
  double f_inv = 1.0 / f;

  for (int i = 0; i < (int)proj.size(); ++i)
  {
    double Z = proj[i].z;
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
	cv::copyMakeBorder(masktmp,masktmp,1,1,1,1,cv::BORDER_REPLICATE);

	std::vector<cv::Mat> diffChannels;
	cv::split(color, diffChannels);

	cv::threshold(diffChannels[2], masktmp, 230, 255, CV_THRESH_BINARY);
	mask = masktmp;
	std::cout << "size mask = " << mask.cols << ", " << mask.rows << std::endl;
	cv::namedWindow("diff",1);
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
											CV_RGB(200, 100, 255) };

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
			index = index > 4? (index % 5) : index;
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
		axis[i].x = (vertices[i].x + vertices[(i+1)%4].x)/2;
		axis[i].y = (vertices[i].y + vertices[(i+1)%4].y)/2;
	}

//	cv::line(dst, axis[0], axis[2], cv::Scalar(255,0,0));
	cv::line(dst, axis[1], axis[3], cv::Scalar(255,0,0));
	index++;
}


bool isFish(cv::linemod::Match& mHead, cv::linemod::Match& mTail, int num_modalities, float threshold, cv::Mat& dst,
	          cv::Ptr<cv::linemod::Detector>& detHead,
		      cv::Ptr<cv::linemod::Detector>& detTail)
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
//		cv::line(dst, axis[i][0], axis[i][2], cv::Scalar(255,0,0));
//		cv::line(dst, axis[i][1], axis[i][3], cv::Scalar(255,0,0));

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



    float omega[4] = {0.3, 0.1, 0.3, 0.3}; //omega parameter
    float f[4];
    float F;
    float alpha[3] = {0.3, 0.3, 0.4};
    float Final;
    float angleL   = 40;   // angle between template LEFT
    float angleR   = 140;  // angle between template RIGHT
    float dist_k   = 80;   // ref distance between template corners
    int   scale    = 3;
    int   overlap_k  = 20;
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
	 double sin_teta = (pt_2.x - pt_1.x)/L;
	 double cos_teta = (pt_2.y - pt_1.y)/L;

	 double tetarad = (pt_2.x - pt_1.x) == 0? 0 : atan((pt_2.y - pt_1.y)/(pt_2.x - pt_1.x));
//	 double tetarad = atan2 ((pt_2.y - pt_1.y),(pt_2.x - pt_1.x));
	 teta = tetarad * 180 / cv_pi;
//	 printf ("The arc tangent for (x=%f, y=%f) is %f degrees\n", (pt_2.x - pt_1.x), (pt_2.x - pt_1.x), tetadeg );

	 speed.val[0] = -1*(k/b)*S*sin_teta;
	 speed.val[1] = -1*(k/b)*S*cos_teta;

	 magSpeed = cv::norm(speed, cv::NORM_L2);

	 return magSpeed;
 }


