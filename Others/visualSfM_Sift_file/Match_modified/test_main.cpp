// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FeaturePoints.h"

using namespace std;

int main()
{   
    int out;
    string save_folder = "C:/Users/gr050/Desktop/sift/";
    string read_folder = "C:/Users/gr050/Desktop/sift_txt/";
    string end_txt = ".txt";
    string end_sift = ".sift";
    string xxx = "C:/Users/gr050/Desktop/TEST/test.txt";

    bool loc_des = true;
    FeatureData test;
    for (int i = 0; i < 371; i++) {
        string name = to_string(i);
        string tmp_save_ = save_folder + name + end_sift;
        const char* tmp_save = tmp_save_.c_str();
        //test.ReadLocFromText(xxx);
        test.ReadLocFromText(read_folder + name + end_txt);
        //test.print_test();
        test.saveSIFTB2(tmp_save);
    }



    /*FeatureData test;
    const char* sift_file= "C:/Users/gr050/Desktop/sift/2.sift";
    out = test.ReadSIFTB(sift_file);

    test.print_test();*/

    return 0;

}
