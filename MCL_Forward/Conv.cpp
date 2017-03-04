#include"Conv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace tbb;

//TODO: 1.ʹ��TBB�����Ż� ��
//      2.������������Ϊָ����������ܷ����� Ӱ�첻��




//void Conv::forward(const std::vector<CnnLayer*>& structure){
//  //�Ƕ��߳��Ż��汾
//	//this->weightҪ�洢ת�ò�Flip���ֵ
//  this->result.clear();
//	vector<Mat> input = structure[this->parents[0]]->result;
//	for (int output_num = 0; output_num < this->dim[0]; output_num++){
//		Mat tmp(input[0].rows - (this->dim[2]) + 1, input[0].cols - (this->dim[3]) + 1, CV_32F);
//		tmp.setTo(this->bias.at<float>(output_num,0));
//		Range r_a(this->dim[2] / 2, this->dim[2] / 2 + tmp.rows);
//		Range r_b(this->dim[3] / 2, this->dim[3] / 2 + tmp.cols);
//		for (int input_num = 0; input_num < this->dim[1]; input_num++){
//			Mat tmp_conv;
//			filter2D(input[input_num], tmp_conv, input[input_num].depth(), this->weight[output_num][input_num]);
//			Mat a= tmp_conv(r_a,r_b);
//			cv::add(tmp, a, tmp);
//		}
//		this->result.push_back(tmp);
//		
//	}
//}

class Parallel_conv : public cv::ParallelLoopBody
{

private:
	const vector<Mat>& inImages;
	const Mat& biasMat;
	const vector<vector<Mat>>& weight;
	const int* dim;
	vector<Mat>& outImages;

public:
	Parallel_conv(const vector<Mat>& inputImgage, const Mat& bias, const vector<vector<Mat>>& weight, vector<Mat>& outImage, const int* dim)
		: inImages(inputImgage), outImages(outImage), biasMat(bias), weight(weight), dim(dim){}

	virtual void operator()(const cv::Range& range) const
	{
		for (int output_num = range.start; output_num < range.end; output_num++)
		{
			Mat tmp(inImages[0].rows - (this->dim[2]) + 1, inImages[0].cols - (this->dim[3]) + 1, CV_32F);
			tmp.setTo(biasMat.at<float>(output_num, 0));
			Range r_a(this->dim[2] / 2, this->dim[2] / 2 + tmp.rows);
			Range r_b(this->dim[3] / 2, this->dim[3] / 2 + tmp.cols);
			for (int input_num = 0; input_num < this->dim[1]; input_num++){
				Mat tmp_conv;
				filter2D(inImages[input_num], tmp_conv, inImages[input_num].depth(), weight[output_num][input_num]);
				cv::add(tmp, tmp_conv(r_a, r_b), tmp);
			}
			outImages[output_num] = tmp;
			tmp.release();
		}
	}
};


void Conv::forward(const std::vector<CnnLayer*>& structure){
	//���߳��Ż��汾
	this->result.clear();
	this->result.resize(this->dim[0]);//Ԥ�ȷ���ռ䣬���Բ��е����������
	cv::parallel_for_(cv::Range(0, this->dim[0]), Parallel_conv(structure[this->parents[0]]->result, this->bias, this->weight, this->result, this->dim));
}