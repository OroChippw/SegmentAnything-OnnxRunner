// SAMOnnxRunner.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include "SAMOnnxRunner.h"

int main()
{
    std::string model_path = "";
    // 
    SAMOnnxRunner Segmentator;
    Segmentator.setSegThreshold(0.95);
    
    std::cout << "Hello World!\n";
}


