#include<iostream>
#include "ncnn/cpu.h"
#include "ncnn/net.h"
#include "ncnn/gpu.h"
#include <ncnn/net.h>
#include<ncnn/mat.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<sys/stat.h>

void listDir(cv::String dir,std::vector<cv::String>& fileLists, std::string item=".jpg")
{
	cv::String testDatadir = "D:/MyNAS/multi_class/data/test_data";
	cv::glob(testDatadir, fileLists,true);

}
int test()
{
	ncnn::Net model;
	//model.opt.use_vulkan_compute = true; // Add this line to enable gpu inference 程序结束后遇到错误
	model.load_param("D:/MyNAS/multi_class/convert_model/multi_class_classification.param");
	model.load_model("D:/MyNAS/multi_class/convert_model/multi_class_classification.bin");
	model.opt.use_vulkan_compute = true;
    
	model.set_vulkan_device(0); // Add this line to enable gpu inference

	const int detector_size_width = 448;
	const int detector_size_height = 448;
	const std::string COLOR[] = { "white", "red", "blue", "black" };
	const std::string TYPE[] = { "dress", "jeans", "shirt", "shoe", "bag" };

	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> timeDuration = endTime - startTime;
	printf("spent time %4f",timeDuration.count());


	cv::String testDatadir = "D:/MyNAS/multi_class/data/test_data/";
	std::vector<cv::String> fileListNames;
	listDir(testDatadir, fileListNames);
	auto start_time  = std::chrono::high_resolution_clock::now();
	int a = 0;
	for (cv::String imgPath : fileListNames)
	{
		a += 1;
		auto startTime = std::chrono::high_resolution_clock::now();
		cv::Mat readMat = cv::imread(imgPath);
		int index = imgPath.rfind("\\");
		imgPath = imgPath.substr(0, index);
		
		index = imgPath.rfind("\\");
		imgPath = imgPath.substr(index, imgPath.size());
		index = imgPath.rfind("_");
		std::string gtColor = imgPath.substr(1, index-1);
		std::string gtTYPE = imgPath.substr(index+1, imgPath.size());
		
		//cv::Mat matInput = readMat.clone();
		int image_width = readMat.cols;
		int image_height = readMat.rows;
		ncnn::Mat ncnnMatInput = ncnn::Mat::from_pixels_resize(readMat.data, ncnn::Mat::PixelType::PIXEL_BGR,
			image_width,image_height, detector_size_width, detector_size_width);
		
		const float mean[3] = { 0.f, 0.f, 0.f };
		const float norm[3] = { 1 / 255.f,1 / 255.f,1 / 255.f };

		ncnnMatInput.substract_mean_normalize(mean, norm);

		ncnn::Extractor ext = model.create_extractor();
		ext.set_num_threads(4);
		ext.input("Input_Image", ncnnMatInput);
		ncnn::Mat out;
		ext.extract("Predict", out);

		float pred_color = 0; 
		int pred_color_index = 0;
		float pred_type = 0; 
		int pred_type_index = 0;

		for (int i = 0; i < out.h; i++)
		{
			const float* values = out.row(i);
			for (int j = 0; j < out.w; j++)
			{
				if (j < 4)
				{
					if (pred_color < values[j])
					{
						pred_color = values[j];
						pred_color_index = j;
					}
				}
				else {
					if (pred_type < values[j])
					{
						pred_type = values[j];
						pred_type_index = j - 4;
					}
				}
			}
		}
		auto endTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> timeDuration = endTime - startTime;

		char textCOLOR[256];
		char textTYPE[256];
		sprintf_s(textCOLOR, "%s : %3f ", COLOR[pred_color_index], pred_color);
		sprintf_s(textTYPE, "%s : %3f ", TYPE[pred_type_index], pred_type);
		int baseLine = 0;
		cv::putText(readMat, textCOLOR, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
		cv::putText(readMat,  textTYPE, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
		cv::imwrite("test/"+std::to_string(a)+".png", readMat);
		printf("use time %4f , color %s <-> %s , type %s <-> %s\n ", timeDuration.count(), gtColor, COLOR[pred_color_index],
			gtTYPE, TYPE[pred_type_index]);
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	timeDuration = end_time - start_time;
	printf("total num %d , use time %4f \n ", fileListNames.size(), timeDuration.count());
	model.clear();
	return 0;
}
int main()
{
	test();
	return 0;
}
