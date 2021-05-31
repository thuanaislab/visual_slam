// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FeaturePoints.h"

using namespace std;

int main()
{   
    int out;
    const char* file = "c:\\users\\gr050\\desktop\\test_jpg\\1311868171.063438.sift";
    const char* file_out = "c:\\users\\gr050\\desktop\\TEST\\TEST.txt";
    bool loc_des = false;
    FeatureData test;
    out = test.ReadSIFTB(file);
    if (out){
        std::cout << "load successful!" << std::endl;
    }
    //out = test.write_out_data(file_out, loc_des);
    //FeatureData::LocationData *location_data;

    return 0;

}
