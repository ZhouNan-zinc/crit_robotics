#include <sys/io.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>

void check_dir(std::string dir){
    if(access(dir.c_str(), F_OK)== -1 && dir != "./"){
        std::cout << "dir not exist, create it" << std::endl;
        int flag = mkdir(dir.c_str(), 0777);
        if(flag == -1){
            std::cout << "mkdir failed" << std::endl;
            throw std::exception();
        }else{
            std::cout << "mkdir success" << std::endl;
        }
    }else{
        std::cout << "dir exist" << std::endl;
    }
}

int main(){
    std::string test_dir = "/home/ghoc/WS/test";
    check_dir(test_dir);
    return 0;
}