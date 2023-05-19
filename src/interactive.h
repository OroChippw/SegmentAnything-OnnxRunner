#pragma once
#include <opencv2/opencv.hpp>
#include "SAMOnnxRunner.h"


void GetClick_handler(int event, int x, int y, int flags, void* data)
{
    ClickInfo* clickinfo = reinterpret_cast<ClickInfo*>(data);
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
    {
        std::cout << "=> Mouse handler trigger : EVENT_LBUTTONDOWN" << std::endl;
        (*clickinfo).pt.x = x;
        (*clickinfo).pt.y = y;
        (*clickinfo).positive = 1;
        break;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
        std::cout << "=> Mouse handler trigger : EVENT_RBUTTONDOWN" << std::endl;
        (*clickinfo).pt.x = x;
        (*clickinfo).pt.y = y;
        (*clickinfo).positive = 0;
        break;
    }

    default:
        break;
    }
}


void GetBox_handler(int event , int x , int y , int flags , void* data)
{
    BoxInfo* boxinfo = reinterpret_cast<BoxInfo*>(data);
    cv::Point left_top;
    cv::Point right_bot;
    switch (event)
    {
    case (cv::EVENT_LBUTTONDOWN):
    {
        left_top.x = x;
        left_top.y = y;
        break;
    }
    case (cv::EVENT_MOUSEMOVE && cv::EVENT_LBUTTONUP):
    {
        right_bot.x = x;
        right_bot.y = y;
        break;
    }
    /*case ():
    {
        right_bot.x;

        break;
    }*/
    default:
        break;
    }
    if (left_top.x > 0 && left_top.y > 0 && right_bot.x > 0 && right_bot.y)
    {
        boxinfo->left_top = left_top;
        boxinfo->right_bot = right_bot;
    }
}