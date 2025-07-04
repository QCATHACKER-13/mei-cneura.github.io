#include <iostream>
#include <string>

using namespace std;

class Person{
    public:
    string first;
    string last;

    void printfulName(){cout << first << " " << last << endl;}
};

int main(){
    Person p;

    p.first = "Raiden";
    p.last = "Mei";

    p.printfulName();
    return 0;
}