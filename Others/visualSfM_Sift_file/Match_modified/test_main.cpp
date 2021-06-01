// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FeaturePoints.h"

using namespace std;

int main()
{   
    int out;
    const char* file = "c:\\users\\gr050\\desktop\\test_jpg\\1311868171.063438.sift";
    const char* file_out = "c:\\users\\gr050\\desktop\\TEST\\TEST____.txt";
    const char* file_txt = "C:/Users/gr050/Desktop/TEST/test.txt";
    const char* save_Bsift = "c:\\users\\gr050\\desktop\\TEST\\bi_sift.sift";
    bool loc_des = true;
    FeatureData test;

    out = test.ReadSIFTB(save_Bsift);
    if (out){
        std::cout << "load successful!" << std::endl;
    }
    out = test.write_out_data(file_out, loc_des);

    //out = test.ReadLocFromText(file_txt);
    //test.print_test();
    //test.saveSIFTB2(save_Bsift);


    return 0;

}
