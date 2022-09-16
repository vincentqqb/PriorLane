/*************************************************************************
	> File Name: evaluate.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: 2016年07月14日 星期四 18时28分45秒
 ************************************************************************/

#include "counter.hpp"
#include "spline.hpp"
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;

void help(void)
{
	cout<<"./evaluate [OPTIONS]"<<endl;
	cout<<"-h                  : print usage help"<<endl;
	cout<<"-a                  : directory for annotation files (default: /data/driving/eval_data/anno_label/)"<<endl;
	cout<<"-d                  : directory for detection files (default: /data/driving/eval_data/predict_label/)"<<endl;
	cout<<"-i                  : directory for image files (default: /data/driving/eval_data/img/)"<<endl;
	cout<<"-l                  : list of images used for evaluation (default: /data/driving/eval_data/img/all.txt)"<<endl;
	cout<<"-w                  : width of the lanes (default: 10)"<<endl;
	cout<<"-t                  : threshold of iou (default: 0.4)"<<endl;
	cout<<"-c                  : cols (max image width) (default: 1920)"<<endl;
	cout<<"-r                  : rows (max image height) (default: 1080)"<<endl;
	cout<<"-s                  : show visualization"<<endl;
	cout<<"-f                  : start frame in the test set (default: 1)"<<endl;
}


void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes);
void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane);
void worker_func(vector<string> &lines_list_v, int start, int end, int &tp, int &fp, int &fn);
void update_tp_fp_fn(int &tp, int &fp, int &fn, int _tp, int _fp, int _fn);

double get_precision(int tp, int fp, int fn)
{
	cerr<<"tp: "<<tp<<" fp: "<<fp<<" fn: "<<fn<<endl;
	if(tp+fp == 0)
	{
		cerr<<"no positive detection"<<endl;
		return -1;
	}
	return tp/double(tp + fp);
}

double get_recall(int tp, int fp, int fn)
{
	if(tp+fn == 0)
	{
		cerr<<"no ground truth positive"<<endl;
		return -1;
	}
	return tp/double(tp + fn);
}

mutex myMutex; 
string anno_dir = "/data/driving/eval_data/anno_label/";
string detect_dir = "/data/driving/eval_data/predict_label/";
string im_dir = "/data/driving/eval_data/img/";
string list_im_file = "/data/driving/eval_data/img/all.txt";
string output_file = "./output.txt";
int width_lane = 10;
double iou_threshold = 0.4;
int im_width = 1920;
int im_height = 1080;
int oc;
bool show = false;
int frame = 1;
int NUM_PROCESS=20;

int main(int argc, char **argv)
{
	// process params
	
	while((oc = getopt(argc, argv, "ha:d:i:l:w:t:c:r:sf:o:p:")) != -1)
	{
		switch(oc)
		{
			case 'h':
				help();
				return 0;
			case 'a':
				anno_dir = optarg;
				break;
			case 'd':
				detect_dir = optarg;
				break;
			case 'i':
				im_dir = optarg;
				break;
			case 'l':
				list_im_file = optarg;
				break;
			case 'w':
				width_lane = atoi(optarg);
				break;
			case 't':
				iou_threshold = atof(optarg);
				break;
			case 'c':
				im_width = atoi(optarg);
				break;
			case 'r':
				im_height = atoi(optarg);
				break;
			case 's':
				show = true;
				break;
			case 'f':
				frame = atoi(optarg);
				break;
			case 'o':
				output_file = optarg;
				break;
			case 'p':
			    NUM_PROCESS = atoi(optarg);
			    break;
		}
	}


	cerr<<"------------Configuration---------"<<endl;
	cerr << "using multi-thread, num:" << NUM_PROCESS << endl;
	cerr<<"anno_dir: "<<anno_dir<<endl;
	cerr<<"detect_dir: "<<detect_dir<<endl;
	cerr<<"im_dir: "<<im_dir<<endl;
	cerr<<"list_im_file: "<<list_im_file<<endl;
	cerr<<"width_lane: "<<width_lane<<endl;
	cerr<<"iou_threshold: "<<iou_threshold<<endl;
	cerr<<"im_width: "<<im_width<<endl;
	cerr<<"im_height: "<<im_height<<endl;
	cerr<<"-----------------------------------"<<endl;
	cerr<<"Evaluating the results..."<<endl;
	// this is the max_width and max_height

	if(width_lane<1)
	{
		cerr<<"width_lane must be positive"<<endl;
		help();
		return 1;
	}


	ifstream ifs_im_list(list_im_file, ios::in);
	if(ifs_im_list.fail())
	{
		cerr<<"Error: file "<<list_im_file<<" not exist!"<<endl;
		return 1;
	}


	vector<string> lines_list_v;
	string line;
	while(getline(ifs_im_list, line)) {
		lines_list_v.push_back(line);
	}
	ifs_im_list.close();

	int TP=0, FP=0, FN=0; //result
	int NUM = lines_list_v.size();
    int batch_size = NUM / NUM_PROCESS;
	vector<thread> thread_v;
	for (int i=0; i<NUM_PROCESS; i++){
			int _start=batch_size*i, _end= batch_size*(i+1);
			_end = (_end>NUM) ? NUM:_end;
			thread_v.push_back(thread(worker_func, ref(lines_list_v), _start, _end, ref(TP), ref(FP), ref(FN)));
	}

	for (int i=0; i<thread_v.size(); i++)
        thread_v[i].join();


	// Counter counter(im_width, im_height, iou_threshold, width_lane);
	
	// vector<int> anno_match;
	// string sub_im_name;
	// int count = 0;


	// while(getline(ifs_im_list, sub_im_name))
	// {
	// 	count++;
	// 	if (count < frame)
	// 		continue;
	// 	string full_im_name = im_dir + sub_im_name;
	// 	string sub_txt_name =  sub_im_name.substr(0, sub_im_name.find_last_of(".")) + ".lines.txt";
	// 	string anno_file_name = anno_dir + sub_txt_name;
	// 	string detect_file_name = detect_dir + sub_txt_name;
	// 	vector<vector<Point2f> > anno_lanes;
	// 	vector<vector<Point2f> > detect_lanes;
	// 	read_lane_file(anno_file_name, anno_lanes);
	// 	read_lane_file(detect_file_name, detect_lanes);
	// 	//cerr<<count<<": "<<full_im_name<<endl;
	// 	anno_match = counter.count_im_pair(anno_lanes, detect_lanes);
	// 	if (show)
	// 	{
	// 		visualize(full_im_name, anno_lanes, detect_lanes, anno_match, width_lane);
	// 		waitKey(0);
	// 	}
	// }
	// ifs_im_list.close();

	cerr << "list images num: " << lines_list_v.size() << endl;
	
	double precision = get_precision(TP, FP, FN);
	double recall = get_recall(TP, FP, FN);
	double F = 2 * precision * recall / (precision + recall);	
	cerr<<"finished process file"<<endl;
	cerr<<"precision: "<<precision<<endl;
	cerr<<"recall: "<<recall<<endl;
	cerr<<"Fmeasure: "<<F<<endl;
	cerr<<"----------------------------------"<<endl;

	ofstream ofs_out_file;
	ofs_out_file.open(output_file, ios::out);
	ofs_out_file<<"file: "<<output_file<<endl;
	ofs_out_file<<"tp: "<< TP <<" fp: "<< FP <<" fn: "<< FN <<endl;
	ofs_out_file<<"precision: "<<precision<<endl;
	ofs_out_file<<"recall: "<<recall<<endl;
	ofs_out_file<<"Fmeasure: "<<F<<endl<<endl;
	ofs_out_file.close();
	return 0;
}

void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes)
{
	lanes.clear();
	ifstream ifs_lane(file_name, ios::in);
	if(ifs_lane.fail())
	{
		return;
	}

	string str_line;
	while(getline(ifs_lane, str_line))
	{
		vector<Point2f> curr_lane;
		stringstream ss;
		ss<<str_line;
		double x,y;
		while(ss>>x>>y)
		{
			curr_lane.push_back(Point2f(x, y));
		}
		lanes.push_back(curr_lane);
	}

	ifs_lane.close();
}

void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane)
{
	Mat img = imread(full_im_name, 1);
	Mat img2 = imread(full_im_name, 1);
	vector<Point2f> curr_lane;
	vector<Point2f> p_interp;
	Spline splineSolver;
	Scalar color_B = Scalar(255, 0, 0);
	Scalar color_G = Scalar(0, 255, 0);
	Scalar color_R = Scalar(0, 0, 255);
	Scalar color_P = Scalar(255, 0, 255);
	Scalar color;
	for (int i=0; i<anno_lanes.size(); i++)
	{
		curr_lane = anno_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		if (anno_match[i] >= 0)
		{
			color = color_G;
		}
		else
		{
			color = color_G;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			cv::line(img, p_interp[n], p_interp[n+1], color, width_lane);
			cv::line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	bool detected;
	for (int i=0; i<detect_lanes.size(); i++)
	{
		detected = false;
		curr_lane = detect_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		for (int n=0; n<anno_lanes.size(); n++)
		{
			if (anno_match[n] == i)
			{
				detected = true;
				break;
			}
		}
		if (detected == true)
		{
			color = color_B;
		}
		else
		{
			color = color_R;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			cv::line(img, p_interp[n], p_interp[n+1], color, width_lane);
			cv::line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	namedWindow("visualize", 1);
	imshow("visualize", img);
	namedWindow("visualize2", 1);
	imshow("visualize2", img2);
}

void update_tp_fp_fn(int &tp, int &fp, int &fn, int _tp, int _fp, int _fn)
{
	std::lock_guard<std::mutex> guard(myMutex);
	tp += _tp;
	fp += _fp;
	fn += _fn;
}

void worker_func(vector<string> &lines_list_v, int start, int end, int &tp, int &fp, int &fn)
{
	Counter counter(im_width, im_height, iou_threshold, width_lane);

	vector<int> anno_match;
	string sub_im_name;
	int count = 0;

	for (int i=start; i<end; i++) {
		sub_im_name = lines_list_v[i];
		count++;

		string full_im_name = im_dir + sub_im_name;
		string sub_txt_name =  sub_im_name.substr(0, sub_im_name.find_last_of(".")) + ".lines.txt";
		string anno_file_name = anno_dir + sub_txt_name;
		string detect_file_name = detect_dir + sub_txt_name;
		vector<vector<Point2f> > anno_lanes;
		vector<vector<Point2f> > detect_lanes;

		read_lane_file(anno_file_name, ref(anno_lanes));
		read_lane_file(detect_file_name, ref(detect_lanes));
		
		anno_match = counter.count_im_pair(ref(anno_lanes), ref(detect_lanes));
	}

	update_tp_fp_fn(ref(tp), ref(fp), ref(fn), counter.getTP(), counter.getFP(), counter.getFN());
}