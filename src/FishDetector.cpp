#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <boost/filesystem.hpp>
#include <cstdio>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
#ifdef WITH_MATLABIO
	#include "MatlabIOModel.hpp"
#endif
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"

#include "functions.hpp"
#include "ConfigFile.h"
#include "Mouse.hpp"
#include "Timer.hpp"


ConfigFile config("detector.conf");


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
		"\t-fn follow by list image filename.\n"    << std::endl;
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

int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;



int main(int argc, char * argv[])
{
  // Various settings and flags
  bool show_match_result = true;
  bool show_timings      = true;
  bool show_matches      = true;
  bool learn_online      = false;
  bool savedlmHead       = false;
  bool savedlmTail       = false;
  bool delaying          = false;
  bool press_key_a       = false;
//  bool maskloaded = false;
//  int maskside = NONE;
  char key;
  cv::vector<int> num_classes;
//  int matching_threshold = 88;
  int matching_threshold_head = config.read<int>("matching_threshold_head");
  int matching_threshold_tail = config.read<int>("matching_threshold_tail");
  int matching_threshold_fish = config.read<int>("matching_threshold_fish");
  int nrCandidate             = config.read<int>("nrCandidate");
  float boxFactor             = config.read<int>("boxFactor");


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
#ifdef WITH_MATLABIO
  else if (ext.compare(".mat") == 0) {
	  model.reset(new MatlabIOModel);
  }
#endif
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
  Timer matchLM_timer;
  Timer matchPB_timer;

  // Initialize HighGUI
  help();
  cv::namedWindow("color");
  cv::namedWindow("sunday");
  cv::namedWindow("connected");
  cv::namedWindow("Mask");
  cv::namedWindow("quantized");

  Mouse::start("color");

  //set video parameter
  const std::string video = "output.avi";   // Form the new name with container

  cv::VideoWriter outputVideo;                                        // Open the output
  outputVideo.open(video, CV_FOURCC('M','J','P','G'), 5, cv::Size(1920,480), true);

  if (!outputVideo.isOpened())
  {
      std::cout  << "Could not open the output video for write: " <<  video.c_str() << std::endl;
      return -1;
  }


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

	  //Read image from imagelist
	  cv::Mat test = cv::imread(imagelist[j], CV_LOAD_IMAGE_COLOR); // do grayscale processing?
	  std::cout << "///" << imagelist[j] << "///" << std::endl;
	  cv::Mat_<float> depth; //Added for PartBasesModel

	  //Resize image to limite process time
	  if((test.rows > 480) || (test.cols > 640) )
	  {
		  cv::resize(test, test, cv::Size(640,480));
	  }

	  cv::imshow("sunday", test);
	  color = test.clone();

	  std::vector<cv::Mat> sources;
	  sources.push_back(color);
	  cv::Mat display = color.clone();
	  cv::Mat display2 = color.clone();

	  // detect potential candidates in the image
	  matchPB_timer.start();
	  std::vector<Candidate> candidates;
	  pbd.detect(color, depth, candidates);
	  printf("Number of candidates: %ld\n", candidates.size());
	  matchPB_timer.stop();

	  // display the best candidates
	  Visualize visualize(model->name());
	  SearchSpacePruning<float> ssp;
	  cv::Mat canvas;
	  if (candidates.size() > 0)
	  {
		  Candidate::sort(candidates);
		  Candidate::nonMaximaSuppression(color, candidates, 0.3);
		  visualize.candidates(color, candidates, nrCandidate, 1,canvas, true);
		  std::cout << "candidate = " << candidates.size() << std::endl;
		  visualize.image(canvas);
	  }

	  // draw each candidate to the canvas
	  cv::Mat maskPB;
	  std::vector<cv::Rect> boundingBoxes;
	  Candidate::mask(color, candidates, nrCandidate, boxFactor, maskPB, boundingBoxes);
	  std::vector<cv::Mat> maskLinemod;
	  maskLinemod.push_back(255*maskPB);
	  cv::Mat masked;
	  color.copyTo(masked, maskLinemod[0]);
	  cv::imshow("Mask", masked);

	  // Perform matching
	  std::vector< cv::linemod::Match > matchesHead;
	  std::vector< cv::linemod::Match > matchesTail;
	  std::vector< std::string        > class_idsHead;
	  std::vector< std::string        > class_idsTail;
	  std::vector< cv::Mat            > quantized_imagesHead;
	  std::vector< cv::Mat            > quantized_imagesTail;
	  matchLM_timer.start();
	  detHead->match(sources, (float)matching_threshold_head, matchesHead,
			  class_idsHead, quantized_imagesHead, maskLinemod);
	  detTail->match(sources, (float)matching_threshold_tail, matchesTail,
			  class_idsTail, quantized_imagesTail, maskLinemod);

	  cv::vector<std::vector<cv::linemod::Match> > matches;
	  matches.push_back(matchesHead);
	  matches.push_back(matchesTail);
	  cv::vector<std::vector<std::string> > class_ids;
	  class_ids.push_back(class_idsHead);
	  class_ids.push_back(class_idsTail);
	  cv::vector<std::vector<cv::Mat> > quantized_images;
	  quantized_images.push_back(quantized_imagesHead);
	  quantized_images.push_back(quantized_imagesTail);

	  cv::Mat colored = displayQuantized(quantized_imagesHead[0]);
	  cv::imshow("quantized", colored);
	  matchLM_timer.stop();

	  cv::vector<int> classes_visited;
	  classes_visited.push_back(0);
	  classes_visited.push_back(0);

	  std::set<std::string>  visitedHead;
	  std::set<std::string>  visitedTail;

	  //Reinit counter
	  classes_visited[0] = 0;
	  classes_visited[1] = 0;

	  // go through all head matches, and connecting  head-tail
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int i = 0; (i < (int)matches[0].size()); ++i)
	  {
		  ++classes_visited[0];
		  //go through all Tail matches
#ifdef _OPENMP
#pragma omp parallel for
#endif
		  for (int j = 0; (j < (int)matches[1].size()); ++j)
		  {
			  ++classes_visited[1];
			  //loading matches
			  cv::linemod::Match mHead = matches[0][i];
			  cv::linemod::Match mTail = matches[1][j];


			  //getting templates
			  const std::vector<cv::linemod::Template>& templatesHead = detHead->getTemplates(mHead.class_id, mHead.template_id);
			  const std::vector<cv::linemod::Template>& templatesTail = detTail->getTemplates(mTail.class_id, mTail.template_id);

			  if (show_match_result)
			  {
				  printf("SimilarityHead: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
						  mHead.similarity, mHead.x, mHead.y, mHead.class_id.c_str(), mHead.template_id);
				  printf("SimilarityTail: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
						  mTail.similarity, mTail.x, mTail.y, mTail.class_id.c_str(), mTail.template_id);
			  }

			  // Draw matching template
			  drawResponse(templatesHead, num_modalities, display, cv::Point(mHead.x, mHead.y), detTail->getT(0));
			  drawResponse(templatesTail, num_modalities, display, cv::Point(mTail.x, mTail.y), detTail->getT(0));
			  drawBoxes(boundingBoxes, display, 1/boxFactor);
			  std::ostringstream similarity;
			  similarity.precision(1);
			  similarity << std::fixed << mHead.similarity;
			  cv::putText(display, similarity.str(), cv::Point(mHead.x,mHead.y),
					  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,250), 1, CV_AA);
			  similarity << std::fixed << mTail.similarity;
			  cv::putText(display, similarity.str(), cv::Point(mTail.x,mTail.y),
					  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,250), 1, CV_AA);



			  bool Fish = isFish(mHead, mTail, num_modalities, (float)matching_threshold_fish, display2, detHead, detTail, config);


			  if(Fish)
			  {

				  drawResponse(templatesHead, num_modalities, display2, cv::Point(mHead.x, mHead.y), detHead->getT(0));
				  drawResponse(templatesTail, num_modalities, display2, cv::Point(mTail.x, mTail.y), detTail->getT(0));
				  drawBoxes(boundingBoxes, display2, 1/boxFactor);

//				  cv::putText(display2, mHead.class_id, cv::Point(mHead.x,mHead.y),
//						  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,250), 1, CV_AA);
//				  cv::putText(display2, mTail.class_id, cv::Point(mTail.x,mTail.y),
//						  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,0,250), 1, CV_AA);


				  printf("--------------Connected---------------------\n");
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
		  }
		  //Reinit counter
		  classes_visited[1] = 0;
	  }
	  //Reinit counter
	  classes_visited[0] = 0;

	  std::vector<cv::Mat> images3;
	  images3.push_back(display);
	  images3.push_back(display2);
	  images3.push_back(canvas);
	  cv::Mat im3 = display3(images3);
//	  cv::imshow("display3", im3);
	  //outputVideo.write(res); //save or
	  outputVideo << im3;
//	  std::stringstream outputImage;
//	  outputImage << "src/image";

	  char *path = (char *)malloc(100);
	  sprintf(path,"images/result%.4d.jpg", j);
	  cv::imwrite(path, im3);

    if (show_match_result && matches.empty())
      printf("No matches found...\n");
    if (show_timings)
    {
      printf("Training          : %.2fs\n", extract_timer.time());
      printf("Matching PartBased: %.2fs\n", matchPB_timer.time());
      printf("Matching LineMode : %.2fs\n", matchLM_timer.time());
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

//    if(j == (imagelist.size()-1))
//    	j = 0;

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
        matching_threshold_fish = std::max(matching_threshold_fish - 1, -100);
        printf("New threshold head: %d\n", matching_threshold_fish);
        break;
      case 's':
        // increment threshold fish
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

