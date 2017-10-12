#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <map>
#include <fstream>
#include <assert.h>
#include "bfs.hpp"

using namespace std;

int main(int argc, char **argv) 
{
    vector<unsigned> VF, EF, VB, EB, visF, visB, scc;
    vector<unsigned>::iterator itr;

    const string fn("test.txt");
    /*const string fn("WikiTalk.txt");
    ofstream out("bfs_r.txt");
    if(!out){  
        cout << "Unable to open outfile";  
        exit(1); // terminate with error  
    }*/

    VF.reserve(N + 1);
    EF.reserve(M);
    VB.reserve(N + 1);
    EB.reserve(M);

    Graphpreproc(fn, VF, EF, VB, EB);
    BFS(VF,EF,0,visF);

    for(itr = visF.begin();itr != visF.end();itr++)
        cout<<*itr<<' ';
    cout<<endl;

    /*for(itr = visF.begin();itr != visF.end();itr++)
        out<<*itr<<'/t';
    out<<endl;*/

    return 0;
}

void Graphpreproc(const string filename, vector<unsigned> &VF, vector<unsigned> &EF, vector<unsigned> &VB, vector<unsigned> &EB)
{
    ifstream in_f;
    vector<unsigned> t;
    vector<unsigned>::iterator itr;
    multimap<const unsigned, unsigned> m;
    multimap<const unsigned, unsigned>::iterator mitr;
    unsigned count = 0;
    in_f.open(filename.c_str(), ios::in);
    while (!in_f.eof()){
        string temp, s1, s2;
        stringstream ss1, ss2;
        unsigned t1, t2;
        getline(in_f, temp);
        if (*(temp.begin()) == '#')
            continue;
        s1 = string(temp, 0, temp.find_first_of('\t'));
        s2 = string(temp, temp.find_first_not_of('\t', temp.find_first_of('\t')), temp.find_last_not_of('\t'));
        ss1 << s1;
        ss1 >> t1;
        ss2 << s2;
        ss2 >> t2;
        t.push_back(t1);
        m.insert(make_pair(t2, t1));
        EF.push_back(t2);
    }

    itr = t.begin();
    VF.push_back(0);
    for (int i = 0; i < N - 1; i++){
        while ((itr != t.end()) && (*itr == i)){
            count++;
            itr++;
        }
        count += VF.at(i);
        VF.push_back(count);
        count = 0;
    }
    VF.push_back(M);

    mitr = m.begin();
    VB.push_back(0);
    for (int i = 0; i < N - 1; i++){
        while ((mitr != m.end()) && ((mitr->first) == i)){
            count++;
            mitr++;
        }
        count += VB.at(i);
        VB.push_back(count);
        count = 0;
    }
    VB.push_back(M);
    for (mitr = m.begin(); mitr != m.end(); mitr++){
        EB.push_back(mitr->second);
    }
}
